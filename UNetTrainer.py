import numpy as np
from torch.utils.data import Dataset
import h5pickle as h5py
import torch.nn as nn
import time
import torch


class PatchDataset(Dataset):
    '''
    Characterizes a patch dataset for PyTorch
    '''

    def __init__(self, path, indexes, n_classes, transform=None):
        self.file = h5py.File(path, 'r')
        self.indexes = indexes
        self.n_classes = n_classes
        self.transform = transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        image = self.file[str(index)]['img'][()]
        label = self.file[str(index)]['lbl'][()]

        X = image.reshape((1, *image.shape))
        y = label

        if self.transform:
            X = self.transform(X)
            y = self.transform(y)

        return X, y


class UNetTrainer:
    '''
    Training UNet with saving/loading model
    '''
    def __init__(self, model, optimizer, loaders, max_epochs, loss_criterion=nn.CrossEntropyLoss(),
                 lr=0.0005, current_epoch=0, best_val_epoch=0, loss=None, accuracy=None):
        self.model = model
        self.optimizer = optimizer
        self.loaders = loaders
        self.initial_lr = lr
        self.max_epochs = max_epochs
        self.loss_criterion = loss_criterion
        self.current_epoch = current_epoch
        self.best_val_epoch = best_val_epoch
        if loss is None:
            self.loss = self.initialize_dict()
        else:
            self.loss = loss
        if accuracy is None:
            self.accuracy = self.initialize_dict()
        else:
            self.accuracy = accuracy

    # This loading classmethod creates a new object of this class with parameters partly from the loaded checkpoints
    @classmethod
    def load_checkpoint(cls, filename, model, optimizer, loaders, max_epochs, loss_criterion=nn.CrossEntropyLoss(),
                 lr=0.0005):
        path = 'Networks/' + filename + '.tar'
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['calculate_accuracy']
        best_val_epoch = checkpoint['best_val_epoch']
        print("Checkpoint loaded. Current Epoch: {:d}, Best Validation Loss: {:.3f}".format(epoch, min(loss['val'])))
        return cls(model, optimizer, loaders, max_epochs, loss_criterion=loss_criterion, lr=lr,
                   current_epoch=epoch, best_val_epoch=best_val_epoch, loss=loss, accuracy=accuracy)

    @staticmethod
    def initialize_dict():
        train_val_dict = {
            'train': [],
            'val': []
        }
        return train_val_dict

    def train(self, batch_print=True, verbose_epoch=1):
        start_time = time.time()

        for ep in range(self.max_epochs):
            self.model.train()  # pytorch way to make model trainable. Needed after .eval()
            self.adapt_learn_rate()
            loss, accuracy, total_voxels = None, None, None
            epoch_loss = []  # Unused right now. Only last loss of epoch is stored

            for i, data in enumerate(self.loaders['train']):
                patch_imgs, patch_lbls = data
                if i == 0:
                    total_voxels = np.prod(patch_lbls.shape)  # Returns the nr of voxels in a batch of patch labels/images

                self.optimizer.zero_grad()
                # forward, backward and optimize
                output = self.model(patch_imgs)
                loss = self.loss_criterion(output, patch_lbls)
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss)
                accuracy = self.calculate_accuracy(output, patch_lbls, total_voxels)
                if batch_print:
                    print("Epoch {:d} \t Batch {:d} \t Loss = {:.3f} \t Accuracy = {:.3f}"
                          .format(self.current_epoch, i, loss, accuracy))

            # Saving the last loss and calculate_accuracy
            self.loss['train'].append(loss.item())
            self.accuracy['train'].append(accuracy)

            # Validation
            val_accuracy, val_loss = self.validation()
            self.loss['val'].append(val_loss.item())
            self.accuracy['val'].append(val_accuracy)
            if val_loss == min(self.loss['val']):
                self.save_network("best_model")
                self.best_val_epoch = self.current_epoch

            if ep % verbose_epoch == 0:
                self.epoch_print()
            self.current_epoch += 1
            self.save_network("last_model")  # Implement saving best validation epoch

        print("--- Training Time: {:.3f} seconds ---".format(time.time() - start_time))

    def validation(self):
        val_loss, accuracies = [], []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.loaders['val']):
                patch_imgs, patch_lbls = data
                if i == 0:
                    total_voxels = np.prod(patch_lbls.shape)  # Returns the number of voxels in a batch of patch labels/images
                output = self.model(patch_imgs)
                accuracy = self.calculate_accuracy(output, patch_lbls, total_voxels)
                loss = self.loss_criterion(output, patch_lbls)
                val_loss.append(loss)
                accuracies.append(accuracy)

        return sum(accuracies) / len(accuracies), sum(val_loss) / len(val_loss)
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
            'calculate_accuracy': self.accuracy,
            'best_val_epoch': self.best_val_epoch
        }, path)

    def single_image_forward(self, image, image_size):
        self.model.eval()
        with torch.no_grad():
            image = image.reshape((1, 1, *image_size))
            one_hot_pred = self.model(torch.from_numpy(image))
            _, pred = torch.max(one_hot_pred, 1)
            pred = pred.numpy().reshape(image_size)
        return pred

    @staticmethod
    def calculate_accuracy(output, patch_lbls, total_voxels):
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == patch_lbls).sum().item()  # Number of correct voxel predictions
        return 100. * correct / total_voxels

    def epoch_print(self):
        print("\n------------------------------------------------------")
        print("Current Epoch = {:d}".format(self.current_epoch))
        print("Training: \t\tLoss = {:.3f} \t Accuracy = {:.3f}"
              .format(self.loss['train'][-1], self.accuracy['train'][-1]))
        print("Validation: \tLoss = {:.3f} \t Accuracy = {:.3f}"
              .format(self.loss['val'][-1], self.accuracy['val'][-1]))
        print("Best Epoch = {:d}".format(self.best_val_epoch))
        print("Validation: \tLoss = {:.3f} \t Accuracy = {:.3f}"
              .format(min(self.loss['val']), self.accuracy['val'][self.best_val_epoch]))
        print("------------------------------------------------------\n")






