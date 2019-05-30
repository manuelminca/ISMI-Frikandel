import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import h5pickle as h5py
import matplotlib.pyplot as plt
from model import Modified3DUNet, SimpleModel
from UNetTrainer import PatchDataset, UNetTrainer
import torch
from loss import GeneralizedDiceLoss, WeightedCrossEntropyLoss
import torch.nn as nn


# Creating a main is necessary in windows for multiprocessing, which is used by the dataloader
def main():
    # patches_file = "50_patches_dataset.h5"
    # hf = h5py.File(patches_file, 'r')
    # We obtain a list with all the IDs of the patches
    # all_groups = list(hf)
    # Dividing the dataset into train and validation. Shuffle has to be false otherwise the model might be trained
    # on what was previously validation set and validated on what was previously train set.
    # X_train, X_validation = train_test_split(all_groups, test_size=0.2, shuffle=False)
    # print(X_train, X_validation)

    # for testing
    train_patches_file = "trainexample.h5"
    val_patches_file = train_patches_file
    # train_patches_file = "train_50_patches_dataset.h5"
    # val_patches_file = "val_50_patches_dataset.h5"
    # Loader Parameters
    params = {'batch_size': 2,
              'shuffle': False,
              'num_workers': 1}

    train_dataset = PatchDataset(train_patches_file, n_classes=3)
    val_dataset = PatchDataset(val_patches_file, n_classes=3)
    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **params)

    loaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Model and param
    model = Modified3DUNet(in_channels=1, n_classes=3)
    optimizer = optim.Adam(model.parameters())
    max_epochs = 10

    # Training model from scratch
    # Median foreground percentage = 0.2 (= class 1,2)
    # Median cancer percentage = 0.01 (= class 2)
    # Median pancreas percentage = 0.2 - 0.01 = 0.19 (= class 1)
    # Median background percentage = 1-0.2 = 99.8 (=class 0)
    # [99.8, 0.19, 0.01] => corresponding class weights = [1, 525, 9980]
    # class_weights = torch.tensor([1., 525., 9980.])
    # loss_criterion = GeneralizedDiceLoss(weight=class_weights)
    # loss_criterion = WeightedCrossEntropyLoss(weight=class_weights)

    weights = [1, 100, 500]
    class_weights = torch.FloatTensor(weights)

    loss_criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainer = UNetTrainer(model, optimizer, loaders, max_epochs, loss_criterion=loss_criterion)
    trainer.train()
    # trainer = UNetTrainer(model, optimizer, loaders, max_epochs)
    # trainer.train()

    # Load from last epoch
    # checkpoint_trainer = UNetTrainer.load_checkpoint("last_model", model, optimizer, loaders, max_epochs)
    # checkpoint_trainer.train()


if __name__ == '__main__':
    main()