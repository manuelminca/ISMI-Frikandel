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
    correct, loss = 0, 0
    total = 1
    for i, data in enumerate(val_loader):
        images, labels = data
        if total == 1:
            for s in labels.shape:
                total *= s
        ans = model(images)
        _, predicted = torch.max(ans.data, 1)
        correct += (predicted == labels).sum().item()
        loss += criterion(ans, labels)
    accuracy = 100. * correct / total
    return accuracy, loss


def train(model, train_loader, val_loader):
    epochs = 2
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    validate = True
    for epoch in range(epochs):
        total_loss = 0.0
        n_correct, n_total = 0, 0
        for i, data in enumerate(train_loader):
            images, labels = data

            model.train()
            optimizer.zero_grad()
            # forward, backward and optimize
            pred = model(images)

            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss
            print("Epoch {:d} \t Batch {:d} \t total_loss = {:.3f}".format(epoch, i, total_loss))

        if validate:
            model.eval()
            with torch.no_grad():
                val_accuracy, val_loss = validation(model, val_loader, criterion)
                print("Validation accuracy = {:.2f} \t Validation loss = {:.3f}".format(val_accuracy, val_loss))
    total_val_loss = 0

    print("Done")
    print("--- Time: {:.3f} seconds ---".format(time.time() - start_time))


# Creating a main is necessary in windows for multiprocessing, which is used by the dataloader
def main():
    patches_file = "patches_dataset_small.h5"
    hf = h5py.File(patches_file, 'r')
    # We obtain a list with all the IDs of the patches
    all_groups = list(hf)
    # Dividing the dataset into train and validation
    X_train, X_validation = train_test_split(all_groups, test_size=0.2)

    # Parameters
    params = {'batch_size': 2,
              'shuffle': False,
              'num_workers': 1}

    train_dataset = PatchDataset(patches_file, X_train, n_classes=3)
    val_dataset = PatchDataset(patches_file, X_validation, n_classes=3)
    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **params)

    # Try to obtain summary of the 3D U-Net
    model = Modified3DUNet(in_channels=1, n_classes=3)
    # summary(model, (1, 256, 256, 32))

    # model = SimpleModel(out_classes=3)
    # summary(model, (1, 64, 64, 16))
    train(model, train_loader, val_loader)


if __name__ == '__main__':
    main()

##############################################################################
## Manuel stuff
##############################################################################

# Generators
#
# # https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader

#
# validation_set = DatasetLoader(patches_file, X_validation)
# validation_generator = data.DataLoader(validation_set, **params)
#
# # print(validation_set[1])
#
# train_iter = iter(train_loader)


# images, labels = train_iter.__next__()
#
# print(images.shape)

# for img, lbl in training_generator:
# 	print(img.shape)


# keys = []
# with h5py.File("patches_dataset.h5", 'r') as f: # open file
#     patch_number = 0
#     all_groups = list(f)
#
#     batch = f[all_groups[patch_number]]
#
#     for i in range(16):
#
#         #print(f[all_groups[str(patch_number+1)]['img']].shape)
#
#         #batch = np.append(batch, f[all_groups[patch_number +1]['img']], axis=0)
#         patch_number += 1
#         #print(batch.shape)
#


#
# # Loading the model
# in_channels = 4
# n_classes = 3
# base_n_filter = 16
# model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
#
#
#
# x_train = torch.from_numpy(x_train)
# y_train = torch.from_numpy(y_train)
#
# batch_size = 32
# dataset = torch.utils.data.TensorDataset(x_train, y_train)
# loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
#
# model = Net()
# opt = optim.Adam(model.parameters())
#
#
# if use_cuda:
#         torch.backends.cudnn.benchmark = True
#         model.cuda()
#
#     # measure
#
#     for epoch in range(2):
#         if epoch == 1:
#             start = time.time()
#         for i, data in enumerate(loader):
#             opt.zero_grad()
#
#             batch, label = data
#             batch = Variable(batch)
#             label = Variable(label)
#
#             if use_cuda:
#                 batch = batch.cuda()
#                 label = label.cuda()
#
#             out = model.forward(batch)
#
#             loss = -out.gather(1, label).log().mean()
#             loss.backward()
#             opt.step()
#
#             if verbose:
#                 print(f"{i}: {loss.data[0]}", end=" "*16+"\r")
#         if epoch == 1:
#             print(f"Elapsed time: {time.time()-start}")
#
#
#
