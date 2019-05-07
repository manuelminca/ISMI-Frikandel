import torch.nn as nn
import torch
import argparse
import os
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import h5py

import numpy as np

class Modified3DUNet(nn.Module):
	def __init__(self, in_channels, n_classes, base_n_filter = 8):
		super(Modified3DUNet, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.base_n_filter = base_n_filter

		self.lrelu = nn.LeakyReLU()
		self.dropout3d = nn.Dropout3d(p=0.6)
		self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
		self.softmax = nn.Softmax(dim=1)

		# Level 1 context pathway
		self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
		self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

		# Level 2 context pathway
		self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
		self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter*2)

		# Level 3 context pathway
		self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
		self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter*4)

		# Level 4 context pathway
		self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
		self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)

		# Level 5 context pathway, level 0 localization pathway
		self.conv3d_c5 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
		self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*8)

		self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
		self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

		# Level 1 localization pathway
		self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
		self.conv3d_l1 = nn.Conv3d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)

		# Level 2 localization pathway
		self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
		self.conv3d_l2 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*2)

		# Level 3 localization pathway
		self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
		self.conv3d_l3 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter)

		# Level 4 localization pathway
		self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
		self.conv3d_l4 = nn.Conv3d(self.base_n_filter*2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

		self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
		self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter*4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)




	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def norm_lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2, mode='nearest'),
			# should be feat_in*2 or feat_in
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def forward(self, x):
		#  Level 1 context pathway
		out = self.conv3d_c1_1(x)
		residual_1 = out
		out = self.lrelu(out)
		out = self.conv3d_c1_2(out)
		out = self.dropout3d(out)
		out = self.lrelu_conv_c1(out)
		# Element Wise Summation
		out += residual_1
		context_1 = self.lrelu(out)
		out = self.inorm3d_c1(out)
		out = self.lrelu(out)

		# Level 2 context pathway
		out = self.conv3d_c2(out)
		residual_2 = out
		out = self.norm_lrelu_conv_c2(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c2(out)
		out += residual_2
		out = self.inorm3d_c2(out)
		out = self.lrelu(out)
		context_2 = out

		# Level 3 context pathway
		out = self.conv3d_c3(out)
		residual_3 = out
		out = self.norm_lrelu_conv_c3(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c3(out)
		out += residual_3
		out = self.inorm3d_c3(out)
		out = self.lrelu(out)
		context_3 = out

		# Level 4 context pathway
		out = self.conv3d_c4(out)
		residual_4 = out
		out = self.norm_lrelu_conv_c4(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c4(out)
		out += residual_4
		out = self.inorm3d_c4(out)
		out = self.lrelu(out)
		context_4 = out

		# Level 5
		out = self.conv3d_c5(out)
		residual_5 = out
		out = self.norm_lrelu_conv_c5(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c5(out)
		out += residual_5
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

		out = self.conv3d_l0(out)
		out = self.inorm3d_l0(out)
		out = self.lrelu(out)

		# Level 1 localization pathway
		out = torch.cat([out, context_4], dim=1)
		out = self.conv_norm_lrelu_l1(out)
		out = self.conv3d_l1(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

		# Level 2 localization pathway
		out = torch.cat([out, context_3], dim=1)
		out = self.conv_norm_lrelu_l2(out)
		ds2 = out
		out = self.conv3d_l2(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

		# Level 3 localization pathway
		out = torch.cat([out, context_2], dim=1)
		out = self.conv_norm_lrelu_l3(out)
		ds3 = out
		out = self.conv3d_l3(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

		# Level 4 localization pathway
		out = torch.cat([out, context_1], dim=1)
		out = self.conv_norm_lrelu_l4(out)
		out_pred = self.conv3d_l4(out)

		ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
		ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
		ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
		ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
		ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

		out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
		seg_layer = out
		out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
		#out = out.view(-1, self.n_classes)
		out = self.softmax(out)
		return out, seg_layer

    # def getBatch(self, dataset):
    #     while True:
    #
    #     keys = []
    #     with h5py.File(path, 'r') as f: # open file
    #         f.visit(keys.append) # append all keys to list
    #         for key in keys:
    #             if ':' in key: # contains data if ':' in key
    #                 print(f[key].name)
    #                 weights[f[key].name] = f[key].value
    #     return weights



hf = h5py.File("patches_dataset.h5", 'r')
#We obtain a list with all the IDs of the patches
all_groups = list(hf)
#Randomly shuffle the patches
#np.random.shuffle(all_groups)
#Dividing the dataset into train and test
X_train, X_validation = train_test_split(all_groups, test_size=0.2)

print(X_train)

class Dataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, path, indexes):
		'Initialization'
		self.file = h5py.File(path, 'r')
		self.indexes = indexes

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.indexes)

	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		X = self.file[str(index)]['img'].value
		y = self.file[str(index)]['lbl'].value

		return X, y


# Parameters
params = {'batch_size': 16,
          'shuffle': False,
          'num_workers': 1}


# Generators

#https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
training_set = Dataset("patches_dataset.h5", X_train)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset("patches_dataset.h5", X_validation)
validation_generator = data.DataLoader(validation_set, **params)


#print(validation_set[1])

train_iter = iter(training_set)

images, labels = train_iter.next()

print(images.shape)

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
