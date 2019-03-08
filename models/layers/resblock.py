"""
This code is the implementation of residual block for GAN.
This is implemented with reference to following links.
- https://github.com/pfnet-research/sngan_projection
- https://github.com/igul222/improved_wgan_training
"""

import torch
from torch import nn
from torch.nn import functional as F

from .conditional_batchnorm import ConditionalBatchNorm2d
from .layernorm import LayerNorm


def NormLayer(type, num_features, num_classes=-1, affine=True):
	if type == 'batchnorm':
		return nn.BatchNorm2d(num_features, affine=affine)
	elif type == 'layernorm':
		return LayerNorm(num_features, affine=affine)
	elif type == 'conditional_batchnorm':
		if num_classes == -1:
			raise ValueError('expected positive value (got -1)')
		return ConditionalBatchNorm2d(num_features, num_classes, affine=affine)


def ActLayer(type, negative_slope=0.2):
	if type == 'relu':
		return nn.ReLU(inplace=True)
	elif type == 'leakyrelu':
		return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)


class Conv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, he_init=True):
		super(Conv, self).__init__()
		self.he_init = he_init
		if kernel_size == 1:
			self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
		elif kernel_size == 3:
			self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)

	def forward(self, x):
		return self.conv(x)


class DownConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, he_init=True):
		super(DownConv, self).__init__()
		self.conv = nn.Sequential(
			nn.AvgPool2d(kernel_size=2),
			Conv(in_channels, out_channels, kernel_size, bias, he_init)
		)

	def forward(self, x):
		return self.conv(x)


class ConvDown(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, he_init=True):
		super(ConvDown, self).__init__()
		self.conv = nn.Sequential(
			Conv(in_channels, out_channels, kernel_size, bias, he_init),
			nn.AvgPool2d(kernel_size=2)
		)

	def forward(self, x):
		return self.conv(x)


class UpConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, he_init=True):
		super(UpConv, self).__init__()
		self.conv = Conv(in_channels, out_channels, kernel_size, bias, he_init)

	def forward(self, x):
		return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, norm_type, act_type, resample=None, num_classes=-1):
		super(ResidualBlock, self).__init__()
		if resample not in [None, 'up', 'down']:
			raise ValueError('resample method is None, "up", or "down" (got {})'.format(resample))
		self.num_classes = num_classes
		self.condition = (num_classes >= 0)
		self.need_shortcut_conv = (in_channels != out_channels) or (resample is not None)

		# shortcut convolution
		if resample == 'up':
			self.shortcut_conv = UpConv(in_channels, out_channels, 1, he_init=False)
		elif resample == 'down':
			self.shortcut_conv = ConvDown(in_channels, out_channels, 1, he_init=False)
		elif resample == None and self.need_shortcut_conv:
			self.shortcut_conv = Conv(in_channels, out_channels, 1, he_init=False)

		# convolution
		if resample == 'up':
			self.conv1 = UpConv(in_channels, out_channels, 3)
			self.conv2 = Conv(out_channels, out_channels, 3)
		elif resample == 'down':
			self.conv1 = Conv(in_channels, in_channels, 3)
			self.conv2 = ConvDown(in_channels, out_channels, 3)
		elif resample is None:
			self.conv1 = Conv(in_channels, in_channels, 3)
			self.conv2 = Conv(in_channels, out_channels, 3)

		# activation
		self.act1 = ActLayer(act_type)
		self.act2 = ActLayer(act_type)

		# normalization
		self.norm1 = NormLayer(norm_type, in_channels, num_classes)
		if resample == 'up':
			self.norm2 = NormLayer(norm_type, out_channels, num_classes)
		else:
			self.norm2 = NormLayer(norm_type, in_channels, num_classes)

	def forward(self, x, t=None):
		o = self.norm1(x, t) if self.condition else self.norm1(x)
		o = self.act1(o)
		o = self.conv1(o)
		o = self.norm2(o, t) if self.condition else self.norm2(o)
		o = self.act2(o)
		o = self.conv2(o)

		return o + self.shortcut(x)

	def shortcut(self, x):
		if self.need_shortcut_conv:
			return self.shortcut_conv(x)
		else:
			return x


class OptimizedBlock(nn.Module):
	def __init__(self, in_channels, out_channels, norm_type, act_type):
		super(OptimizedBlock, self).__init__()
		self.conv = nn.Sequential(
			Conv(in_channels, out_channels, 3),
			ActLayer(act_type),
			ConvDown(out_channels, out_channels, 3)
		)
		self.shortcut = DownConv(in_channels, out_channels, 1, he_init=False)

	def forward(self, x):
		return self.conv(x) + self.shortcut(x)
