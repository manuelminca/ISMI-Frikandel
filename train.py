import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import h5pickle as h5py
import torch.nn as nn
from model import Modified3DUNet
from torchsummary import summary
# from Create_Dataset import Pancreas, PatchExtractor


class PatchDataset(Dataset):
    '''
    Characterizes a patch dataset for PyTorch
    '''

    def __init__(self, path, indexes, transform=None):
        self.file = h5py.File(path, 'r')
        self.indexes = indexes
        self.transform = transform  # Could be used for upsampling, normalisation and toTensor. Although dataloader already does toTensor so might not be needed
        #  For multiple transforms check torchvision.transforms.Compose

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        X = self.file[str(index)]['img'][()]
        y = self.file[str(index)]['lbl'][()]

        if self.transform:
            X = self.transform(X)
            y = self.transform(y)

        return X, y


class SimpleModel(nn.Module):
    '''
    Simple self made 3D U-net model for testing purposes. The conv layers have same in and
    output shape (excluding channels). Similar to padding=same in tensorflow
    '''

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv3d_b1_1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_b1_2 = nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.max_pool_1 = nn.MaxPool3d((2,2,1))

        self.conv3d_b2_1 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_b2_2 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.max_pool_2 = nn.MaxPool3d((2,2,1))

        self.conv3d_b3_1 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_b3_2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample_1 = nn.Upsample(scale_factor=(2,2,1))

        self.conv3d_b4_1 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_b4_2 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample_2 = nn.Upsample(scale_factor=(2,2,1))

        self.conv3d_b5_1 = nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_b5_2 = nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv3d_end = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv3d_b1_1(x)
        residual_1 = x
        x = self.conv3d_b1_2(x)
        x = self.max_pool_1(x)

        x = self.conv3d_b2_1(x)
        residual_2 = x
        x = self.conv3d_b2_2(x)
        x = self.max_pool_2(x)

        x = self.conv3d_b3_1(x)
        x = self.conv3d_b3_2(x)
        x = self.upsample_1(x)

        # x = torch.cat([x, residual_2], dim=1)
        x = self.conv3d_b4_1(x)
        x = self.conv3d_b4_2(x)
        x = self.upsample_2(x)

        # x = torch.cat([x, residual_1], dim=1)
        x = self.conv3d_b5_1(x)
        x = self.conv3d_b5_2(x)

        out = self.conv3d_end(x)
        return out


# Creating a main is necessary in windows for multiprocessing, which is used by the dataloader
def main():

    patches_file = "patches_dataset_short.h5"
    hf = h5py.File(patches_file, 'r')
    # We obtain a list with all the IDs of the patches
    all_groups = list(hf)
    # Dividing the dataset into train and validation
    X_train, X_validation = train_test_split(all_groups, test_size=0.2)


    # Parameters
    params = {'batch_size': 2,
              'shuffle': False,
              'num_workers': 1}

    train_dataset = PatchDataset(patches_file, X_train)
    train_loader = DataLoader(train_dataset, **params)

    train_iter = iter(train_loader)

    # # Testing
    # images, labels = train_iter.next()
    # print('images shape on batch size = {}'.format(images.size()))
    # print('labels shape on batch size = {}'.format(labels.size()))

    # Try to obtain summary of the 3D U-Net
    # model = Modified3DUNet(in_channels=1, n_classes=3)
    # summary(model, (1, 256, 256, 30))

    model = SimpleModel()
    summary(model, (1, 256, 256, 30))



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
