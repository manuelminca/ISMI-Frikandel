import numpy as np
from torch.utils.data import Dataset
# import h5pickle as h5py
import torch.nn as nn
import h5py
import time
import torch
from loss import compute_per_channel_dice


class PatchDataset(Dataset):
    '''
    Characterizes a patch dataset for PyTorch
    '''

    def __init__(self, path, n_classes, transform=None):
        self.file = h5py.File(path, 'r')
        self.n_classes = n_classes
        self.transform = transform

    def __len__(self):
        return len(list(self.file))

    def __getitem__(self, index):
        image = self.file[str(index)]['img'][()]
        label = self.file[str(index)]['lbl'][()]

        # Getting right shapes and types for pytorch
        X = image.reshape((1, *image.shape)).astype(np.float32)
        y = label.astype(np.int64)

        if self.transform:
            X = self.transform(X)
            y = self.transform(y)

        return X, y


class UNetTrainer:
    '''
    Training UNet with saving/loading model
    '''
    def __init__(self, model, optimizer, loaders, max_epochs, device="cpu", loss_criterion=nn.CrossEntropyLoss(),
                 lr=0.0005, current_epoch=0, best_val_epoch=0, batch_size=2, loss=None,
                 accuracy=None, dice=None, name=""):
        self.model = model
        self.optimizer = optimizer
        self.loaders = loaders
        self.initial_lr = lr
        self.max_epochs = max_epochs
        self.device = torch.device(device)
        self.loss_criterion = loss_criterion
        self.current_epoch = current_epoch
        self.best_val_epoch = best_val_epoch
        self.batch_size = batch_size
        self.name = name
        self.classes = 3
        self.time = 0
        if loss is None:
            self.loss = self.initialize_dict()
        else:
            self.loss = loss
        if accuracy is None:
            self.accuracy = self.initialize_dict()
        else:
            self.accuracy = accuracy
        if dice is None:
            self.dice = self.initialize_dict()
        else:
            self.dice = dice

    # This loading classmethod creates a new object of this class with parameters partly from the loaded checkpoints
    @classmethod
    def load_checkpoint(cls, filename, model, optimizer, loaders, max_epochs, device="cpu",
                        loss_criterion=nn.CrossEntropyLoss(), lr=0.0005):
        path = 'Networks/' + filename + '.tar'
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        dice = checkpoint['dice']
        best_val_epoch = checkpoint['best_val_epoch']
        name = checkpoint['name']
        print("Checkpoint loaded. Current Epoch: {:d}, Best Validation Loss: {:.3f}".format(epoch, min(loss['val'])))
        return cls(model, optimizer, loaders, max_epochs, device=device, loss_criterion=loss_criterion, lr=lr,
                   current_epoch=epoch, best_val_epoch=best_val_epoch, loss=loss, accuracy=accuracy, dice=dice, name=name)

    @staticmethod
    def initialize_dict():
        train_val_dict = {
            'train': [],
            'val': []
        }
        return train_val_dict

    def train(self, batch_print=True, verbose_batch=1):
        start_time = time.time()

        for ep in range(self.max_epochs):
            self.model.train()  # pytorch way to make model trainable. Needed after .eval()
            self.adapt_learn_rate()
            total_voxels, loss, accuracy, dice = None, None, None, None
            dices, epoch_loss = [], []
            epoch_start = time.time()

            for i, data in enumerate(self.loaders['train']):
                patch_imgs, patch_lbls = data
                patch_imgs, patch_lbls = patch_imgs.to(self.device), patch_lbls.to(self.device)
                if i == 0:
                    total_voxels = np.prod(patch_lbls.shape)  # Returns the nr of voxels in a batch of patch labels/images

                self.optimizer.zero_grad()
                # forward, backward and optimize
                output = self.model(patch_imgs)
                loss = self.loss_criterion(output, patch_lbls)
                loss.backward()
                self.optimizer.step()

                # epoch_loss.append(loss.item())
                accuracy = self.calculate_accuracy(output, patch_lbls, total_voxels)
                dice = self.calculate_dice(output, patch_lbls)
                dices.append(dice)
                epoch_loss.append(loss.item())

                if batch_print and (i % verbose_batch == 0):
                    print("Epoch {:d}   Batch {:d}   Loss = {:.3f}   Accuracy = {:.2f}   Dice = {:.2f} {:.2f} {:.2f}"
                          .format(self.current_epoch, i, loss, accuracy, *dice))

            # Saving the last loss, accuracy and dice
            self.loss['train'].append(self.avg(epoch_loss))
            self.accuracy['train'].append(accuracy)
            self.dice['train'].append(self.avg(dices))

            # Validation
            val_accuracy, val_loss, val_dice = self.validation()
            self.loss['val'].append(val_loss.item())
            self.accuracy['val'].append(val_accuracy)
            self.dice['val'].append(val_dice)
            if val_loss == min(self.loss['val']):
                self.save_network(self.name + "best_model")
                self.best_val_epoch = self.current_epoch

            self.epoch_print(time.time() - epoch_start)
            self.current_epoch += 1
            self.save_network(self.name + "last_model")  # Implement saving best validation epoch

        print("--- Training Time: {:.3f} seconds ---".format(time.time() - start_time))

    def validation(self):
        val_loss, accuracies, dice = [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.loaders['val']):
                patch_imgs, patch_lbls = data
                patch_imgs, patch_lbls = patch_imgs.to(self.device), patch_lbls.to(self.device)
                if i == 0:
                    total_voxels = np.prod(patch_lbls.shape)  # Returns the number of voxels in a batch of patch labels/images
                output = self.model(patch_imgs)
                accuracies.append(self.calculate_accuracy(output, patch_lbls, total_voxels))
                dice.append(self.calculate_dice(output, patch_lbls))
                val_loss.append(self.loss_criterion(output, patch_lbls))

        return self.avg(accuracies), self.avg(val_loss), self.avg(dice)
        # Returning avg loss and calculate_accuracy over all batches

    def adapt_learn_rate(self):
        # Exponentially decrease learning rate with each epoch
        lr = self.initial_lr * 0.985 ** self.current_epoch
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save_network(self, filename, checkpoint=True):
        if checkpoint:
            path = 'Networks/' + filename + '.tar'
        else:
            path = 'Networks/' + filename + '.pt'
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'accuracy': self.accuracy,
            'dice': self.dice,
            'best_val_epoch': self.best_val_epoch,
            'name': self.name
        }, path)

    def single_image_forward(self, image):
        self.model.eval()
        with torch.no_grad():
            print(image.shape)
            image = image.reshape((1, *image.shape))
            one_hot_pred = self.model(torch.from_numpy(image).to(self.device))
            _, pred = torch.max(one_hot_pred, 1)
            pred = pred.cpu().numpy().reshape((image.shape[2], image.shape[3], image.shape[4]))
        return pred

    def calculate_dice(self, output, patch_lbls):
        # patch_lbls_one_hot = torch.FloatTensor(output.size())
        patch_lbls_one_hot = torch.cuda.FloatTensor(output.size())
        patch_lbls_one_hot.zero_()
        one_hot_shape = (patch_lbls.size()[0], 1, patch_lbls.size()[1], patch_lbls.size()[2], patch_lbls.size()[3])
        patch_lbls_one_hot.scatter_(1, patch_lbls.view(one_hot_shape), 1)
        return compute_per_channel_dice(output, patch_lbls_one_hot).data.cpu().numpy()

    @staticmethod
    def calculate_accuracy(output, patch_lbls, total_voxels):
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == patch_lbls).sum().item()  # Number of correct voxel predictions
        return 100. * correct / total_voxels

    @staticmethod
    def avg(list):
        return sum(list)/len(list)

    def epoch_print(self, epoch_time):
        print("\n------------------------------------------------------")
        print("Current Epoch = {:d}".format(self.current_epoch))
        print("Training:   \tLoss = {:.3f}   Accuracy = {:.2f}   Dice = {:.2f} {:.2f} {:.2f}"
              .format(self.loss['train'][-1], self.accuracy['train'][-1], *self.dice['train'][-1]))
        print("Validation: \tLoss = {:.3f}   Accuracy = {:.2f}   Dice = {:.2f} {:.2f} {:.2f}"
              .format(self.loss['val'][-1], self.accuracy['val'][-1], *self.dice['val'][-1]))
        print("Best Epoch = {:d}".format(self.best_val_epoch))
        print("Validation: \tLoss = {:.3f}   Accuracy = {:.2f}   Dice = {:.2f} {:.2f} {:.2f}"
              .format(min(self.loss['val']), self.accuracy['val'][self.best_val_epoch], *self.dice['val'][self.best_val_epoch]))
        print("--- Epoch Time: {:.3f} Seconds ---".format(epoch_time))
        print("------------------------------------------------------\n")
