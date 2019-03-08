"""
LeNet
	For details, please refer to the original paper (http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).
"""

import sys
import torch
from torch import nn


__all__ = [
	'LeNet', 
	'lenet'
]


def convblock(in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=True):
	"""
	Returns convolution block
	"""
	if use_bn:
		return [
			nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True)
		]
	else:
		return [
			nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
			nn.ReLU(True)
		]


class LeNet(nn.Module):
	"""LeNet model
	Original paper is "Gradient-Based Learning Applied to Document Recognition"

	Args:
		num_classes (int): number of classes
		use_bn (bool): if True, use batch normalization layer
	"""
	def __init__(self, num_classes, use_bn=True):
		super(LeNet, self).__init__()

		self.features = nn.Sequential(
			*convblock(in_channels=3, out_channels=6, kernel_size=5, use_bn=use_bn),
			nn.MaxPool2d(kernel_size=2),
			*convblock(in_channels=6, out_channels=16, kernel_size=5, use_bn=use_bn),
			nn.MaxPool2d(kernel_size=2)
		)
		self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
		self.classifier = nn.Sequential(
			nn.Linear(16*5*5, 128),
			nn.ReLU(True),
			nn.Linear(128, 84),
			nn.ReLU(True),
			nn.Linear(84, num_classes)
		)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = out.view(x.size(0), -1)
		x = self.classifier(x)
		return x


def lenet(num_classes=100, use_bn=True):
	"""LeNet model
	pre-trained model is not available.

	Args:
		num_classes (int): number of classes
		use_bn (bool): If True, returns a model with batch normalization layer
	"""

	model = LeNet(num_classes, use_bn)
	return model
