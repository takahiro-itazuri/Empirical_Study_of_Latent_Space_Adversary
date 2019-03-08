"""
This code is the implementation of "Generating Natural Adversarial Examples".
(https://openreview.net/pdf?id=H1BLjgZCb)

The original implemention by TensorFlow is available.
(https://github.com/zhengliz/natural-adversary)

This is implemented with reference to following links.
- https://github.com/zhengliz/natural-adversary
- https://github.com/hjbahng/natural-adversary-pytorch
- https://github.com/igul222/improved_wgan_training
"""

import torch
from torch import nn

from .layers import *
from .wgan_gp import Generator32, Critic32, Generator64, Critic64

__all__ = ['get_wgan_gp_inverter']


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
	def __init__(self, nz, nc, ngf):
		super(Generator28, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ngf = ngf

		self.fc = nn.Sequential(
			nn.Linear(nz, 4 * 4 * 4 * ngf),
			nn.ReLU(inplace=True)
		)
		self.conv = nn.Sequential(
			nn.ConvTranspose2d(4 * ngf, 2 * ngf, kernel_size=5, stride=2, padding=2),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(2 * ngf, 1 * ngf, kernel_size=4, stride=2, padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(1 * ngf, nc, kernel_size=4, stride=2, padding=1),
			nn.Tanh()
		)

	def forward(self, x):
		x = self.fc(x)
		x = x.view(-1, 4 * self.ngf, 4, 4)
		x = self.conv(x)
		return x


# ============================== Critics ============================= #

class Critic28(nn.Module):
	def __init__(self, nc, ncf):
		super(Critic28, self).__init__()
		self.nc = nc
		self.ncf = ncf

		self.conv = nn.Sequential(
			nn.Conv2d(nc, 1 * ncf, kernel_size=5, stride=2, padding=2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(1 * ncf, 2 * ncf, kernel_size=5, stride=2, padding=2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(2 * ncf, 4 * ncf, kernel_size=5, stride=2, padding=2),
			nn.LeakyReLU(0.2, inplace=True)
		)
		self.fc = nn.Sequential(
			nn.Linear(4 * 4 * 4 * ncf, 1)
		)

	def forward(self, x):
		x = self.conv(x)
		x = x.view(-1, 4 * 4 * 4 * self.ncf)
		x = self.fc(x)
		return x


# ============================= Inverters ============================ #

class Inverter28(nn.Module):
	def __init__(self, nz, nc, nif):
		super(Inverter28, self).__init__()
		self.nz = nz
		self.nc = nc
		self.nif = nif

		self.conv = nn.Sequential(
			nn.Conv2d(nc, 1 * nif, kernel_size=5, stride=2, padding=2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(1 * nif, 2 * nif, kernel_size=5, stride=2, padding=2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(2 * nif, 4 * nif, kernel_size=5, stride=2, padding=2),
			nn.LeakyReLU(0.2, inplace=True)
		)
		self.fc = nn.Sequential(
			nn.Linear(4 * 4 * 4 * nif, 8 * nif),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(8 * nif, nz)
		)

	def forward(self, x):
		x = self.conv(x)
		x = x.view(-1, 4 * 4 * 4 * self.nif)
		x = self.fc(x)
		return x


class Inverter32(nn.Module):
	def __init__(self, nz, nc, nif, norm_type='layernorm', act_type='relu'):
		super(Inverter32, self).__init__()
		self.nz = nz
		self.nc = nc
		self.nif = nif

		self.block1 = OptimizedBlock(nc, nif, norm_type, act_type)
		self.block2 = ResidualBlock(nif, nif, norm_type, act_type, 'down')
		self.block3 = ResidualBlock(nif, nif, norm_type, act_type, None)
		self.block4 = ResidualBlock(nif, nif, norm_type, act_type, None)
		self.act = ActLayer(act_type)
		self.fc = nn.Sequential(
			nn.Linear(8 * 8 * nif, 2048),
			ActLayer(act_type),
			nn.Linear(2048, 512),
			ActLayer(act_type),
			nn.Linear(512, nz)
		)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.act(x).view(x.size(0), -1)
		x = self.fc(x)
		return x


class Inverter64(nn.Module):
	def __init__(self, nz, nc, nif, norm_type='layernorm', act_type='relu'):
		super(Inverter64, self).__init__()
		self.nz = nz
		self.nc = nc
		self.nif = nif

		self.block1 = OptimizedBlock(nc, 1 * nif, norm_type, act_type)
		self.block2 = ResidualBlock(1 * nif, 2 * nif, norm_type, act_type, 'down')
		self.block3 = ResidualBlock(2 * nif, 4 * nif, norm_type, act_type, 'down')
		self.block4 = ResidualBlock(4 * nif, 8 * nif, norm_type, act_type, 'down')
		self.block5 = ResidualBlock(8 * nif, 16 * nif, norm_type, act_type, 'down')
		self.act = ActLayer(act_type)
		self.fc = nn.Sequential(
			nn.Linear(2 * 2 * 16 * nif, 2048),
			ActLayer(act_type),
			nn.Linear(2048, 512),
			ActLayer(act_type),
			nn.Linear(512, nz)
		)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x).view(x.size(0), -1)
		x = self.fc(x)
		return x


# ============================ Interface ============================ #

def get_wgan_gp_inverter(nz, nc, nf, image_size):
	"""
	Returns generator, critic and inverter of WGAN-GP

	Args:
	- nz (int): latent variable size
	- nc (int): image channel size
	- nf (int): number of features for generator, critic, inverter
	- image_size (int): image size
	"""

	if image_size == 28:
		G = Generator28(nz, nc, nf)
		C = Critic28(nc, nf)
		I = Inverter28(nz, nc, nf)
	elif image_size == 32:
		G = Generator32(nz, nc, nf, num_classes=-1)
		C = Critic32(nc, nf, num_classes=-1)
		I = Inverter32(nz, nc, nf)
	elif image_size == 64:
		G = Generator64(nz, nc, nf, num_classes=-1)
		C = Critic64(nc, nf, num_classes=-1)
		I = Inverter64(nz, nc, nf)

	initialize_weights(G)
	initialize_weights(C)
	initialize_weights(I)

	return G, C, I
