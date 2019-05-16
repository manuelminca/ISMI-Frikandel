import torch
from model import Modified3DUNet
import torch.nn as nn
import torch.optim as optim


def save_network(model, optimizer, epoch, loss, checkpoint, filename='network_savefile'):
    if checkpoint:
        path = 'Networks/' + filename + '.tar'
    else:
        path = 'Networks/' + filename + '.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_checkpoint(filename):
	path = 'Networks/' + filename + '.tar'
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


