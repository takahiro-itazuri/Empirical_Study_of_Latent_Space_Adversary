import os
import sys
import random

import torch
import torchvision
from torch.utils.data import Subset

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
data_root = os.path.join(base, 'data')


__all__ = [
	'get_labels',
	'get_transform',
	'get_dataset',
	'get_dataset_stats',
	'normalize',
	'unnormalize',
	'shuffle_dataset'
]


# dataset mean and std calculated by https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
means = {
	'mnist'    : [0.13066049],
	'svhn'     : [0.43768210, 0.44376970, 0.47280442],
	'cifar10'  : [0.49139968, 0.48215841, 0.44653091],
	'cifar100' : [0.50707516, 0.48654887, 0.44091784],
	'stl10'    : [0.44671062, 0.43980984, 0.40664645],
	'imagenet' : [0.485, 0.456, 0.406]
}

stds = {
	'mnist'    : [0.30810780],
	'svhn'     : [0.19803012, 0.20101562, 0.19703614],
	'cifar10'  : [0.24703223, 0.24348513, 0.26158784],
	'cifar100' : [0.26733429, 0.25643846, 0.27615047],
	'stl10'    : [0.26034098, 0.25657727, 0.27126738],
	'imagenet' : [0.229, 0.224, 0.225]
}


def check_dataset(name):
	if name in ['mnist', 'svhn', 'cifar10', 'cifar100', 'stl10', 'imagenet']:
		return True
	else:
		return False


def get_dataset_stats(name, device='cpu'):
	if check_dataset(name):
		mean = means[name]
		std = stds[name]
	else:
		mean = [0.5]
		std = [0.5]

	return torch.tensor(mean, dtype=torch.float32).to(device), torch.tensor(std, dtype=torch.float32).to(device)


def normalize(x, name, device='cpu'):
	"""
	Normalize batch samples

	Args:
	x (torch.Tensor): input value (B, C, W, H)
	name (str): dataset name
	device (str): device
	"""
	mean, std = get_dataset_stats(name, device)
	
	x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
	return x


def unnormalize(x, name, device='cpu'):
	"""
	Unnormalize batch samples

	Args:
	x (torch.Tensor): input value (B, C, W, H)
	name (str): dataset name
	device (str): device
	"""
	mean, std = get_dataset_stats(name, device)

	x.mul_(std[None, :, None, None]).add_(mean[None, :, None, None])
	return x


def get_labels(name, root=None):
	return Idx2Label(name, root)


class Idx2Label():
	def __init__(self, name, root=None):
		self.labels = []
		if check_dataset(name):
			with open(os.path.join(base, 'misc/labels/{}.txt'.format(name)), 'r') as f:
				for line in f:
					self.labels.append(line.split('\t')[1].replace('\n', ''))
		elif root != None:
			for f in os.listdir(root):
				if os.path.isdir(os.path.join(root, f)):
					self.labels.append(f)
		else:
			sys.stderr.write('{} does not exist.'.format(name))
			exit()

	def __getitem__(self, idx):
		return self.labels[idx]

	def __len__(self):
		return len(self.labels)


def get_transform(name, train, input_size, normalize, augment):
	transform = []

	if augment:
		size = 256 if input_size == 224 else int(1.15 * input_size)
		if train:
			transform.extend([
				torchvision.transforms.RandomResizedCrop(input_size),
				torchvision.transforms.RandomHorizontalFlip(),
			])
		else:
			transform.extend([
				torchvision.transforms.Resize(256 if input_size == 224 else int(1.15*input_size)),
				torchvision.transforms.CenterCrop(input_size),
			])
	else:
		transform.extend([
			torchvision.transforms.Resize(input_size)
		])
	
	transform.extend([
		torchvision.transforms.ToTensor()
	])

	if normalize:
		mean, std = get_dataset_stats(name)
		transform.extend([
			torchvision.transforms.Normalize(mean=mean, std=std)
		])

	return torchvision.transforms.Compose(transform)


def get_dataset(name, train, input_size=224, normalize=True, augment=True, num_samples=-1):
	root = os.path.join(data_root, name)

	transform = get_transform(name, train, input_size, normalize, augment)

	if name == 'mnist':
		dataset = torchvision.datasets.MNIST(root, train=train, download=True, transform=transform)
	elif name == 'svhn':
		dataset = torchvision.datasets.SVHN(root, split='train' if train else 'test', download=True, trasnform=transform)
	elif name == 'cifar10':
		dataset = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=transform)
	elif name == 'cifar100':
		dataset = torchvision.datasets.CIFAR100(root, train=train, download=True, transform=transform)
	elif name == 'stl10':
		dataset = torchvision.datasets.STL10(root, split='train' if train else 'test', download=True, transform=transform)
	elif name == 'imagenet':
		root = os.path.join(root, 'train' if train else 'val')
		dataset = torchvision.datasets.ImageFolder(root, transform=transform)

	if num_samples != -1:
		num_samples = min(num_samples, len(dataset))
		indices = range(len(dataset))
		indices = random.sample(indices, num_samples)
		dataset = Subset(dataset, indices)

	return dataset


def shuffle_dataset(dataset, seed=0):
	random.seed(seed)
	indices = list(range(len(dataset)))
	random.shuffle(indices)
	return Subset(dataset, indices)
