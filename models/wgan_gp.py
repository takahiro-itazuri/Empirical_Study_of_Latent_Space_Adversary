"""
This code is the implementation of "Improved Training of Wasserstein GANs".
(https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf)

The original implementation by TensorFlow is available.
(https://github.com/igul222/improved_wgan_training)

This is implemented with reference to following links.
- https://github.com/igul222/improved_wgan_training
- https://github.com/ermongroup/generative_adversary
"""

import torch
from torch import nn

from .layers import *

__all__ = ['get_wgan_gp']


def initialize_weights(model):
	for m in model.modules():
		if isinstance(m, Conv):
			if m.he_init:
				nn.init.kaiming_uniform_(m.conv.weight)
			else:
				nn.init.xavier_uniform_(m.conv.weight)
			if m.conv.bias is not None:
				nn.init.constant_(m.conv.bias, 0.0)
		elif isinstance(m, nn.Linear):
			nn.init.xavier_uniform_(m.weight)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0.0)


# ============================ Generators ============================ #

class Generator28(nn.Module):
	def __init__(self, nz, nc, ngf, norm_type='batchnorm', act_type='relu', num_classes=-1):
		super(Generator28, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.num_classes = num_classes
		self.condition = (num_classes >= 0)
		if self.condition:
			norm_type = 'conditional_batchnorm'

		self.fc1 = nn.Linear(nz, 7 * 7 * ngf)
		self.block2 = ResidualBlock(ngf, ngf, norm_type, act_type, 'up', num_classes)
		self.block3 = ResidualBlock(ngf, ngf, norm_type, act_type, 'up', num_classes)
		self.final = nn.Sequential(
			NormLayer('batchnorm', ngf),
			ActLayer(act_type),
			Conv(ngf, nc, 3, he_init=False),
			nn.Tanh()
		)

	def forward(self, z, t=None):
		x = self.fc1(z)
		x = x.view(-1, self.ngf, 7, 7)
		x = self.block2(x, t)
		x = self.block3(x, t)
		x = self.final(x)
		return x


class Generator32(nn.Module):
	def __init__(self, nz, nc, ngf, norm_type='batchnorm', act_type='relu', num_classes=-1):
		super(Generator32, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.num_classes = num_classes
		self.condition = (num_classes >= 0)
		if self.condition:
			norm_type = 'conditional_batchnorm'

		self.fc1 = nn.Linear(nz, 4 * 4 * ngf)
		self.block2 = ResidualBlock(ngf, ngf, norm_type, act_type, 'up', num_classes)
		self.block3 = ResidualBlock(ngf, ngf, norm_type, act_type, 'up', num_classes)
		self.block4 = ResidualBlock(ngf, ngf, norm_type, act_type, 'up', num_classes)
		self.final = nn.Sequential(
			NormLayer('batchnorm', ngf),
			ActLayer(act_type),
			Conv(ngf, nc, 3, he_init=False),
			nn.Tanh()
		)

	def forward(self, z, t=None):
		x = self.fc1(z)
		x = x.view(-1, self.ngf, 4, 4)
		x = self.block2(x, t)
		x = self.block3(x, t)
		x = self.block4(x, t)
		x = self.final(x)
		return x


class Generator64(nn.Module):
	def __init__(self, nz, nc, ngf, norm_type='batchnorm', act_type='relu', num_classes=-1):
		super(Generator64, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.num_classes = num_classes
		self.condition = (num_classes >= 0)
		if self.condition:
			norm_type = 'conditional_batchnorm'

		self.fc1 = nn.Linear(nz, 4 * 4 * 8 * ngf)
		self.block2 = ResidualBlock(8 * ngf, 8 * ngf, norm_type, act_type, 'up', num_classes)
		self.block3 = ResidualBlock(8 * ngf, 4 * ngf, norm_type, act_type, 'up', num_classes)
		self.block4 = ResidualBlock(4 * ngf, 2 * ngf, norm_type, act_type, 'up', num_classes)
		self.block5 = ResidualBlock(2 * ngf, 1 * ngf, norm_type, act_type, 'up', num_classes)
		self.final = nn.Sequential(
			NormLayer('batchnorm', ngf),
			ActLayer(act_type),
			Conv(1 * ngf, nc, 3),
			nn.Tanh()
		)

	def forward(self, z, t=None):
		x = self.fc1(z)
		x = x.view(-1, 8 * self.ngf, 4, 4)
		x = self.block2(x, t)
		x = self.block3(x, t)
		x = self.block4(x, t)
		x = self.block5(x, t)
		x = self.final(x)
		return x


# ============================= Critics ============================= #

class Critic28(nn.Module):
	def __init__(self, nc, ncf, norm_type='layernorm', act_type='relu', num_classes=-1):
		super(Critic28, self).__init__()
		self.nc = nc
		self.ncf = ncf
		self.num_classes = num_classes
		self.condition = (num_classes >= 0)

		self.block1 = OptimizedBlock(nc, ncf, norm_type, act_type)
		self.block2 = ResidualBlock(ncf, ncf, norm_type, act_type, 'down')
		self.block3 = ResidualBlock(ncf, ncf, norm_type, act_type, None)
		self.block4 = ResidualBlock(ncf, ncf, norm_type, act_type, None)
		self.pool = nn.Sequential(
			ActLayer(act_type),
			nn.AvgPool2d(kernel_size=7)
		)
		self.wgan = nn.Linear(ncf, 1)
		if self.condition:
			self.ac = nn.Linear(ncf, num_classes)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.pool(x).view(x.size(0), -1)

		if self.condition:
			return self.wgan(x), self.ac(x)
		else:
			return self.wgan(x)


class Critic32(nn.Module):
	def __init__(self, nc, ncf, norm_type='layernorm', act_type='relu', num_classes=-1):
		super(Critic32, self).__init__()
		self.nc = nc
		self.ncf = ncf
		self.num_classes = num_classes
		self.condition = (num_classes >= 0)

		self.block1 = OptimizedBlock(nc, ncf, norm_type, act_type)
		self.block2 = ResidualBlock(ncf, ncf, norm_type, act_type, 'down')
		self.block3 = ResidualBlock(ncf, ncf, norm_type, act_type, None)
		self.block4 = ResidualBlock(ncf, ncf, norm_type, act_type, None)
		self.pool = nn.Sequential(
			ActLayer(act_type),
			nn.AvgPool2d(kernel_size=8)
		)
		self.wgan = nn.Linear(ncf, 1)
		if self.condition:
			self.ac = nn.Linear(ncf, num_classes)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.pool(x).view(x.size(0), -1)

		if self.condition:
			return self.wgan(x), self.ac(x)
		else:
			return self.wgan(x)


class Critic64(nn.Module):
	def __init__(self, nc, ncf, norm_type='layernorm', act_type='relu', num_classes=-1):
		super(Critic64, self).__init__()
		self.nc = nc
		self.ncf = ncf
		self.num_classes = num_classes
		self.condition = (num_classes >= 0)


		self.block1 = OptimizedBlock(nc, 1 * ncf, norm_type, act_type)
		self.block2 = ResidualBlock(1 * ncf, 2 * ncf, norm_type, act_type, 'down')
		self.block3 = ResidualBlock(2 * ncf, 4 * ncf, norm_type, act_type, 'down')
		self.block4 = ResidualBlock(4 * ncf, 8 * ncf, norm_type, act_type, 'down')
		self.block5 = ResidualBlock(8 * ncf, 16 * ncf, norm_type, act_type, 'down')
		self.pool = nn.Sequential(
			ActLayer(act_type),
			nn.AvgPool2d(kernel_size=2)
		)
		self.wgan = nn.Linear(16 * ncf, 1)
		if self.condition:
			self.ac = nn.Linear(16 * ncf, num_classes)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)
		x = self.pool(x).view(x.size(0), -1)

		if self.condition:
			return self.wgan(x), self.ac(x)
		else:
			return self.wgan(x)


# ============================ Interface ============================ #

def get_wgan_gp(nz, nc, nf, image_size, condition=False, num_classes=-1):
	"""
	Returns generator and critic of WGAN-GP.

	Args:
	- nz (int): latent variable size
	- nc (int): input image size
	- nf (int): number of features for generator and discriminator
	- image_size (int): image size
	- condition (bool): if True, returns conditional WGAN-GP model
	- num_classes (int): number of classes (only need when `condition` is True)
	"""

	if condition:
		_num_classes = num_classes
	else:
		_num_classes = -1

	if image_size == 28:
		G = Generator28(nz, nc, nf, num_classes=_num_classes)
		C = Critic28(nc, nf, num_classes=_num_classes)
	elif image_size == 32:
		G = Generator32(nz, nc, nf, num_classes=_num_classes)
		C = Critic32(nc, nf, num_classes=_num_classes)
	elif image_size == 64:
		G = Generator64(nz, nc, nf, num_classes=_num_classes)
		C = Critic64(nc, nf, num_classes=_num_classes)

	initialize_weights(G)
	initialize_weights(C)

	return G, C
