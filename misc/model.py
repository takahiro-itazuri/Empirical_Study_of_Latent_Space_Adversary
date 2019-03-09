import os
import sys

import torch
import torchvision
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc.models import *


__all__ = [
	'get_classifier',
	'replace_relu_with_softplus'
]


def get_classifier(name, num_classes=1000, pretrained=False, inplace=True, use_bn=True):

	if pretrained:
		if name == 'vgg16':
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
		if name == 'lenet':
			model = lenet(num_classes=num_classes, use_bn=use_bn)
		elif name == 'alexnet':
			model = alexnet_v2(num_classes=num_classes, use_bn=use_bn)
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


def replace_relu_with_softplus(model, beta=100.0, threshold=10.0):
	"""
	Replace all ReLU activations in ``model`` with Softplus activations.

	Args:
	model (nn.Module): model
	"""
	relu_list = []
	for k, m in model.named_modules():
		if isinstance(m, nn.ReLU):
			relu_list.append(k)
	# print(relu_list)

	module = model._modules
	for name in relu_list:
		splited_name = name.split('.')
		for n in splited_name:
			# sys.stdout.write('[{}]'.format(n))
			if n.isdecimal():
				n = int(n)

			if isinstance(module, BasicBlock) or isinstance(module, Bottleneck):
				module.relu = nn.Softplus(beta=beta, threshold=threshold)
				break

			if isinstance(module[n], nn.ReLU):
				module[n] = nn.Softplus(beta=beta, threshold=threshold)
				break

			module = module[n]

		# sys.stdout.write('\n')
		module = model._modules
