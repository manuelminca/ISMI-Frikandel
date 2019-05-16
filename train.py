import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import h5pickle as h5py
import torch.nn as nn
import torch.optim as optim
import time
import torch
from torchvision.transforms import ToTensor
from model import Modified3DUNet, SimpleModel
from torchsummary import summary


class PatchDataset(Dataset):
    '''
    Characterizes a patch dataset for PyTorch
    '''

    def __init__(self, path, indexes, n_classes, transform=None):
        self.file = h5py.File(path, 'r')
        self.indexes = indexes
        self.n_classes = n_classes
        self.transform = transform  # Could be used for upsampling, normalisation and toTensor. Although dataloader already does toTensor so might not be needed
        #  For multiple transforms check torchvision.transforms.Compose

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


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Return accuracy
def validation(model, val_loader, criterion):
    val_loss, accuracies = [], []
    total_voxs = 1
    for i, data in enumerate(val_loader):
        print(i)
        patch_imgs, patch_lbls = data
        if total_voxs == 1:
            for s in patch_lbls.shape:
                total_voxs *= s   # Returns the number of voxels in a batch of patch labels/images
        ans = model(patch_imgs)
        _, predicted = torch.max(ans.data, 1)
        correct = (predicted == patch_lbls).sum().item()  # Number of correct voxel predictions
        loss = criterion(ans, patch_lbls)
        accuracy = 100. * correct / total_voxs
        val_loss.append(loss)
        accuracies.append(accuracy)

    return sum(accuracies)/len(accuracies), sum(val_loss)/len(val_loss)
    #Returning avg loss and accuracy over all batches


def adapt_learn_rate(optimizer, epoch):
    # Use an initial learn rate of 0.0005; slightly decrease lr with 0.0005 * 0.985**epoch
    lr = 0.0005 * 0.985**epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, val_loader):
    epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    start_time = time.time()
    validate = True
    total_loss = []
    for epoch in range(epochs):
        adapt_learn_rate(optimizer, epoch)
        epoch_loss = []
        for i, data in enumerate(train_loader):
            patch_imgs, patch_lbls = data
            if i == 0:
                total_voxs = np.prod(patch_lbls.shape)

            model.train()
            optimizer.zero_grad()
            # forward, backward and optimize
            ans = model(patch_imgs)

            loss = criterion(ans, patch_lbls)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss)
            _, predicted = torch.max(ans.data, 1)
            correct = (predicted == patch_lbls).sum().item()  # Number of correct voxel predictions
            accuracy = 100. * correct / total_voxs
            print("Epoch {:d} \t Batch {:d} \t Loss = {:.3f} \t Accuracy = {:.3f}".format(epoch, i, loss, accuracy))


            loss += criterion(ans, patch_lbls)

        if validate:
            model.eval()
            with torch.no_grad():
                val_accuracy, val_loss = validation(model, val_loader, criterion)
                print("Validation accuracy = {:.2f} \t Validation loss = {:.3f}".format(val_accuracy, val_loss))

        total_loss.append(epoch_loss)
    total_val_loss = 0

    print("Done")
    print("--- Time: {:.3f} seconds ---".format(time.time() - start_time))


# Creating a main is necessary in windows for multiprocessing, which is used by the dataloader
def main():
    patches_file = "patches_dataset_small.h5"
    model_file = saved_network.pt
    hf = h5py.File(patches_file, 'r')
    # We obtain a list with all the IDs of the patches
    all_groups = list(hf)
    # Dividing the dataset into train and validation
    X_train, X_validation = train_test_split(all_groups, test_size=0.2)

    # Parameters
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1}

    train_dataset = PatchDataset(patches_file, X_train, n_classes=3)
    val_dataset = PatchDataset(patches_file, X_validation, n_classes=3)
    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **params)

    # Try to obtain summary of the 3D U-Net
    model = Modified3DUNet(in_channels=1, n_classes=3)
    # summary(model, (1, 256, 256, 32))

    train(model, train_loader, val_loader)


if __name__ == '__main__':
    main()


# https://pytorch.org/tutorials/beginner/saving_loading_models.html
# Set checkpoint to false s
def save_network(model, optimizer, epoch, loss, checkpoint=True):
    if checkpoint:
        path = 'network_checkpoint.tar'
    else:
        path = 'network_save.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_checkpoint(path='network_checkpoint.tar'):
    model = Modified3DUNet(in_channels=1, n_classes=3)
    optimizer = optim.Adam(model.parameters())
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.train()
    return model, optimizer, epoch, loss


# Load a pt network file on a GPU
def load_saved_network_gpu(from_gpu,path='network_save.pt'):
    device = torch.device("cuda")
    model = Modified3DUNet(in_channels=1, n_classes=3)
    optimizer = optim.Adam(model.parameters())

    if from_gpu:
        file = torch.load(path)
    else:
        file = torch.load(path, map_location="cuda:0")
    model.load_state_dict(file['model_state_dict'])
    optimizer.load_state_dict(file['optimizer_state_dict'])
    epoch = file['epoch']
    loss = file['loss']


# Load a pt network file on a CPU
def load_saved_network_cpu(from_gpu=True):
    return True


