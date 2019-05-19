import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import h5pickle as h5py
from torchsummary import summary
from model import Modified3DUNet, SimpleModel
from UNetTrainer import PatchDataset, UNetTrainer


# Creating a main is necessary in windows for multiprocessing, which is used by the dataloader
def main():
    patches_file = "patches_dataset_nn_dim.h5"
    hf = h5py.File(patches_file, 'r')
    # We obtain a list with all the IDs of the patches
    all_groups = list(hf)
    # Dividing the dataset into train and validation. Shuffle has to be false otherwise the model might be trained
    # on what was previously validation set and validated on what was previously train set.
    X_train, X_validation = train_test_split(all_groups, test_size=0.2, shuffle=False)

    print(X_train, X_validation)

    # Loader Parameters
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1}

    train_dataset = PatchDataset(patches_file, X_train, n_classes=3)
    val_dataset = PatchDataset(patches_file, X_validation, n_classes=3)
    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **params)

    loaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Model and param
    model = Modified3DUNet(in_channels=1, n_classes=3)
    optimizer = optim.Adam(model.parameters())
    max_epochs = 2

    # Training model from scratch
    trainer = UNetTrainer(model, optimizer, loaders, max_epochs)
    trainer.train()

    # Load from last epoch
    # checkpoint_trainer = UNetTrainer.load_checkpoint("last_model", model, optimizer, loaders, max_epochs)
    # checkpoint_trainer.train()


if __name__ == '__main__':
    main()