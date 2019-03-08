import torch
import torchvision
from torch import nn


__all__ = [
	'get_classifier'
]


def get_classifier(name, num_classes=1000, pretrained=False, inplace=True):

	if pretrained:
		if name == 'alexnet':
			raise NotImplementedError
		elif name == 'vgg16':
			model = torchvision.models.vgg16(True)
		elif name == 'vgg16_bn':
			model = torchvision.models.vgg16_bn(True)
		elif name == 'vgg19':
			model = torchvision.models.vgg19(True)
		elif name == 'vgg19_bn':
			model = torchvision.models.vgg19_bn(True)
		elif name == 'resnet18':
			model = torchvision.models.resnet18(True)
		elif name == 'resnet34':
			model = torchvision.models.resnet34(True)
		elif name == 'resnet50':
			model = torchvision.models.resnet50(True)
		elif name == 'resnet101':
			model = torchvision.models.resnet101(True)
		elif name == 'resnet152':
			model = torchvision.models.resnet152(True)
		else:
			raise NotImplementedError

		if num_classes != 1000:
			if name == 'alexnet' or name.startswith('vgg'):
				num_features = model.classifier[6].in_features
				model.classifier[6] = nn.Linear(num_features, num_classes)
			elif name.startswith('resnet'):
				num_features = model.fc.in_features
				model.fc = nn.Linear(num_features, num_classes)

	else:
		if name == 'alexnet':
			model = AlexNet_v1(num_classes=num_classes, use_bn=True)
		elif name == 'vgg16':
			model = torchvision.models.vgg16(False, num_classes=num_classes)
		elif name == 'vgg16_bn':
			model = torchvision.models.vgg16_bn(False, num_classes=num_classes)
		elif name == 'vgg19':
			model = torchvision.models.vgg19(False, num_classes=num_classes)
		elif name == 'vgg19_bn':
			model = torchvision.models.vgg19_bn(False, num_classes=num_classes)
		elif name == 'resnet18':
			model = torchvision.models.resnet18(False, num_classes=num_classes)
		elif name == 'resnet34':
			model = torchvision.models.resnet34(False, num_classes=num_classes)
		elif name == 'resnet50':
			model = torchvision.models.resnet50(False, num_classes=num_classes)
		elif name == 'resnet101':
			model = torchvision.models.resnet101(False, num_classes=num_classes)
		elif name == 'resnet152':
			model = torchvision.models.resnet152(False, num_classes=num_classes)
		else:
			raise NotImplementedError

	for m in model.modules():
		if isinstance(m, nn.ReLU):
			m.inplace = inplace

	return model


class ConvBlock(nn.Module):
	"""Convolution Block"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=True):
		super(ConvBlock, self).__init__()

		if use_bn:
			self.conv = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(True)
			)
		else:
			self.conv = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
				nn.ReLU(True)
			)

	def forward(self, x):
		return self.conv(x)


class AlexNet_v1(nn.Module):
	"""AlexNet model (version 1)
	Original paper is "ImageNet Classification with Deep Convolutional Neural Networks" (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
	In this implementation, local response normalization layer is replced by batch normalization layer.

	Args:
		num_classes (int): number of classes
		use_bn (bool): if True, use batch normalization layer
	"""
	def __init__(self, num_classes=1000, use_bn=True):
		super(AlexNet_v1, self).__init__()

		self.features = nn.Sequential(
			ConvBlock(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, use_bn=use_bn),
			nn.MaxPool2d(kernel_size=3, stride=2),
			ConvBlock(in_channels=96, out_channels=256, kernel_size=5, padding=2, use_bn=use_bn),
			nn.MaxPool2d(kernel_size=3, stride=2),
			ConvBlock(in_channels=256, out_channels=384, kernel_size=3, padding=1, use_bn=use_bn),
			ConvBlock(in_channels=384, out_channels=384, kernel_size=3, padding=1, use_bn=use_bn),
			ConvBlock(in_channels=384, out_channels=256, kernel_size=3, padding=1, use_bn=use_bn)
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Linear(4096, num_classes)
		)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.classifier(x)
		return x
