import os
import sys
import random
import urllib
import zipfile
import lmdb
import cv2
import numpy as np

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
	'shuffle_dataset',
	'download_dataset'
]


# dataset mean and std calculated by https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
means = {
	'mnist'    : [0.13066049],
	'svhn'     : [0.43768210, 0.44376970, 0.47280442],
	'cifar10'  : [0.49139968, 0.48215841, 0.44653091],
	'cifar100' : [0.50707516, 0.48654887, 0.44091784],
	'stl10'    : [0.44671062, 0.43980984, 0.40664645],
	'lsun'     : [0.485, 0.456, 0.406], # copied from imagenet
	'imagenet' : [0.485, 0.456, 0.406]
}

stds = {
	'mnist'    : [0.30810780],
	'svhn'     : [0.19803012, 0.20101562, 0.19703614],
	'cifar10'  : [0.24703223, 0.24348513, 0.26158784],
	'cifar100' : [0.26733429, 0.25643846, 0.27615047],
	'stl10'    : [0.26034098, 0.25657727, 0.27126738],
	'lsun'     : [0.485, 0.456, 0.406], # copied from imagenet
	'imagenet' : [0.485, 0.456, 0.406]
}

dataset_list = ['mnist', 'svhn', 'cifar10', 'cifar100', 'stl10', 'lsun', 'imagenet']


def check_dataset(name):
	if name in dataset_list:
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
	
	x.sub(mean[None, :, None, None]).div(std[None, :, None, None])
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

	x.mul(std[None, :, None, None]).add(mean[None, :, None, None])
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
		dataset = torchvision.datasets.SVHN(root, split='train' if train else 'test', download=True, transform=transform)
	elif name == 'cifar10':
		dataset = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=transform)
	elif name == 'cifar100':
		dataset = torchvision.datasets.CIFAR100(root, train=train, download=True, transform=transform)
	elif name == 'stl10':
		dataset = torchvision.datasets.STL10(root, split='train' if train else 'test', download=True, transform=transform)		
	elif name in ['lsun', 'imagenet']:
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


def download_all_datasets():
	for name in dataset_list:
		download_dataset(name)


def download_dataset(name):
	if not check_dataset(name):
		raise NotImplementedError

	root = os.path.join(data_root, name)

	# === MNIST === #
	if name == 'mnist':
		dataset = torchvision.datasets.MNIST(root, train=True, download=True)
		dataset = torchvision.datasets.MNIST(root, train=False, download=True)
		del dataset

	# === SVHN === #
	elif name == 'svhn':
		dataset = torchvision.datasets.SVHN(root, split='train', download=True)
		dataset = torchvision.datasets.SVHN(root, split='test', download=True)
		del dataset

	# === CIFAR-10 === #
	elif name == 'cifar10':
		dataset = torchvision.datasets.CIFAR10(root, train=True, download=True)
		dataset = torchvision.datasets.CIFAR10(root, train=False, download=True)
		del dataset

	# === CIFAR-100 === #
	elif name == 'cifar100':
		dataset = torchvision.datasets.CIFAR100(root, train=True, download=True)
		dataset = torchvision.datasets.CIFAR100(root, train=False, download=True)
		del dataset

	# === LSUN === #
	elif name == 'lsun':
		category_list = ['bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower']

		# download dataset
		os.makedirs(root, exist_ok=True)
		for category in category_list:
			if not os.path.exists(os.path.join(root, '{}_train_lmdb.zip'.format(category))):
				url = 'http://dl.yf.io/lsun/scenes/{}_train_lmdb.zip'.format(category)
				urllib.request.urlretrieve(url, os.path.join(root, '{}_train_lmdb.zip'.format(category)))

		# extract zip file
		for category in category_list:
			if not os.path.exists(os.path.join(root, '{}_train_lmdb'.format(category))):
				with zipfile.ZipFile(os.path.join(root, '{}_train_lmdb.zip'.format(category))) as z:
					z.extractall(root)

		for c in range(len(category_list)):
			os.makedirs(os.path.join(root, 'train', '{:03d}'.format(c)), exist_ok=True)
			os.makedirs(os.path.join(root, 'val', '{:03d}'.format(c)), exist_ok=True)
	
		# save images
		for category_idx, category in enumerate(category_list):
			img_idx = 0

			env = lmdb.open(os.path.join(root, '{}_train_lmdb'.format(category)), max_readers=126, readonly=True)
			with env.begin(write=False) as txn:
				cursor = txn.cursor()
				print('{}'.format(category))

				# train
				for key, val in cursor:
					img = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)
					# img = cv2.resize(img, (256, 256))
					img = cv2.resize(img, (64, 64))
					
					if img_idx < 100000:
						path = os.path.join(root, 'train', '{:03d}'.format(category_idx), '{:08d}.png'.format(category_idx * 100000 + img_idx))					
					elif img_idx < 110000:
						path = os.path.join(root, 'val', '{:03d}'.format(category_idx), '{:08d}.png'.format(category_idx * 10000 + (img_idx - 100000)))
					else:
						break

					cv2.imwrite(path, img)
					img_idx += 1

			del env

