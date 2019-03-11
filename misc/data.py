import os
import sys
import random
import urllib
import zipfile
import tarfile
import shutil
import lmdb
import cv2
import numpy as np
from PIL import Image
from scipy import io

import torch
import torchvision
from torch.utils.data import Subset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

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
	'download_dataset',
	'CustomTensorDataset'
]


# dataset mean and std 
# https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
# https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949
means = {
	'mnist'     : [0.13066049],
	'svhn'      : [0.43768210, 0.44376970, 0.47280442],
	'cifar10'   : [0.49139968, 0.48215841, 0.44653091],
	'cifar100'  : [0.50707516, 0.48654887, 0.44091784],
	'stl10'     : [0.44671062, 0.43980984, 0.40664645],
	'lsun'      : [0.485, 0.456, 0.406], # copied from imagenet
	'cub200'    : [0.47199252, 0.47448921, 0.41031790],
	'dog120'    : [0.46614176, 0.42734158, 0.37495670],
	'food101'   : [0.56214333, 0.42337474, 0.28560150],
	'flower102' : [0.64367676, 0.46372274, 0.40175039],
	'imagenet'  : [0.485, 0.456, 0.406]
}

stds = {
	'mnist'     : [0.30810780],
	'svhn'      : [0.19803012, 0.20101562, 0.19703614],
	'cifar10'   : [0.24703223, 0.24348513, 0.26158784],
	'cifar100'  : [0.26733429, 0.25643846, 0.27615047],
	'stl10'     : [0.26034098, 0.25657727, 0.27126738],
	'lsun'      : [0.229, 0.224, 0.225], # copied from imagenet
	'cub200'    : [0.24212829, 0.23832704, 0.26648488],
	'dog120'    : [0.27039561, 0.26119319, 0.25956830],
	'food101'   : [0.25203338, 0.25410590, 0.24439417],
	'flower102' : [0.26119214, 0.25937790, 0.30645496],
	'imagenet'  : [0.229, 0.224, 0.225]
}

dataset_list = ['mnist', 'svhn', 'cifar10', 'cifar100', 'stl10', 'lsun', 'cub200', 'dog120', 'food101', 'flower102', 'imagenet']


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


# https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949
def calculate_stats(path):
	dataset = ImageFolder(path, transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()]))
	loader = torch.utils.data.DataLoader(dataset, batch_size=100)

	mean = 0.0
	std = 0.0
	for img, _ in loader:
		img = img.view(img.size(0), img.size(1), -1)
		mean += img.mean(dim=2).sum(dim=0)
	mean /= len(dataset)

	var = 0.0
	for img, _ in loader:
		img = img.view(img.size(0), img.size(1), -1)
		var += ((img - mean[None, :, None])**2).mean(dim=2).sum(dim=0)
	std = torch.sqrt(var / len(dataset))

	return mean, std


def normalize(x, name, device='cpu'):
	"""
	Normalize batch samples

	Args:
	x (torch.Tensor): input value (B, C, W, H)
	name (str): dataset name
	device (str): device
	"""
	mean, std = get_dataset_stats(name, device)
	return x.sub(mean[None, :, None, None]).div(std[None, :, None, None])


def unnormalize(x, name, device='cpu'):
	"""
	Unnormalize batch samples

	Args:
	x (torch.Tensor): input value (B, C, W, H)
	name (str): dataset name
	device (str): device
	"""
	mean, std = get_dataset_stats(name, device)
	return 	x.mul(std[None, :, None, None]).add(mean[None, :, None, None])


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
				transforms.RandomResizedCrop(input_size),
				transforms.RandomHorizontalFlip(),
			])
		else:
			transform.extend([
				transforms.Resize(256 if input_size == 224 else int(1.15*input_size)),
				transforms.CenterCrop(input_size),
			])
	else:
		transform.extend([
			transforms.Resize(input_size)
		])
	
	transform.extend([
		transforms.ToTensor()
	])

	if normalize:
		mean, std = get_dataset_stats(name)
		transform.extend([
			transforms.Normalize(mean=mean, std=std)
		])

	return transforms.Compose(transform)


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
	elif name in ['lsun', 'cub200', 'dog102', 'food101', 'imagenet']:
		root = os.path.join(root, 'train' if train else 'val')
		dataset = ImageFolder(root, transform=transform)

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

	# === STL-10 === #
	elif name == 'stl10':
		dataset = torchvision.datasets.STL10(root, split='train', download=True)		
		dataset = torchvision.datasets.STL10(root, split='test', download=True)		
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

	# === CUB-200 === #
	elif name == 'cub200':
		os.makedirs(root, exist_ok=True)

		if not os.path.exists(os.path.join(root, 'images.tgz')):
			url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz'
			urllib.request.urlretrieve(url, os.path.join(root, 'images.tgz'))

		if not os.path.exists(os.path.join(root, 'lists.tgz')):
			url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz'
			urllib.request.urlretrieve(url, os.path.join(root, 'lists.tgz'))

		if not os.path.exists(os.path.join(root, 'images')):
			with tarfile.open(os.path.join(root, 'images.tgz'), 'r:gz') as tf:
				tf.extractall(os.path.join(root))

		if not os.path.exists(os.path.join(root, 'lists')):
			with tarfile.open(os.path.join(root, 'lists.tgz'), 'r:gz') as tf:
				tf.extractall(os.path.join(root))

		# make directories
		os.makedirs(os.path.join(root, 'train'), exist_ok=True)
		os.makedirs(os.path.join(root, 'val'), exist_ok=True)
		for d in os.listdir(os.path.join(root, 'images')):
			if not d.startswith('._'):
				os.makedirs(os.path.join(root, 'train', d), exist_ok=True)
				os.makedirs(os.path.join(root, 'val', d), exist_ok=True)

		with open(os.path.join(root, 'lists/train.txt'), 'r') as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip() # remove new line
				if os.path.exists(os.path.join(root, 'images', line)):
					shutil.move(os.path.join(root, 'images', line), os.path.join(root, 'train', line))

		with open(os.path.join(root, 'lists/test.txt'), 'r') as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip() # remove new line
				if os.path.exists(os.path.join(root, 'images', line)):
					shutil.move(os.path.join(root, 'images', line), os.path.join(root, 'val', line))

		shutil.rmtree(os.path.join(root, 'images'))

	# === Stanford Dog (Dog-120) === #
	elif name == 'dog120':
		os.makedirs(root, exist_ok=True)

		if not os.path.exists(os.path.join(root, 'images.tar')):
			url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
			urllib.request.urlretrieve(url, os.path.join(root, 'images.tar'))

		if not os.path.exists(os.path.join(root, 'lists.tar')):
			url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar'
			urllib.request.urlretrieve(url, os.path.join(root, 'lists.tar'))

		if not os.path.exists(os.path.join(root, 'images')):
			with tarfile.open(os.path.join(root, 'images.tar'), 'r') as tf:
				tf.extractall(os.path.join(root))

		if not os.path.exists(os.path.join(root, 'lists')):
			with tarfile.open(os.path.join(root, 'lists.tar'), 'r') as tf:
				tf.extractall(os.path.join(root, 'lists'))

		# make directories
		os.makedirs(os.path.join(root, 'train'), exist_ok=True)
		os.makedirs(os.path.join(root, 'val'), exist_ok=True)
		for d in os.listdir(os.path.join(root, 'images')):
			if not d.startswith('._'):
				os.makedirs(os.path.join(root, 'train', d), exist_ok=True)
				os.makedirs(os.path.join(root, 'val', d), exist_ok=True)

		matdata = io.loadmat(os.path.join(root, 'lists', 'train_list.mat'), squeeze_me=True)
		for i in range(matdata['file_list'].shape[0]):
			line = matdata['file_list'][i]
			if os.path.exists(os.path.join(root, 'images', line)):
				shutil.move(os.path.join(root, 'images', line), os.path.join(root, 'train', line))

		matdata = io.loadmat(os.path.join(root, 'lists', 'test_list.mat'), squeeze_me=True)
		for i in range(matdata['file_list'].shape[0]):
			line = matdata['file_list'][i]
			if os.path.exists(os.path.join(root, 'images', line)):
				shutil.move(os.path.join(root, 'images', line), os.path.join(root, 'val', line))

		shutil.rmtree(os.path.join(root, 'images'))
		del matdata

	# === food101 === #
	elif name == 'food101':
		os.makedirs(root, exist_ok=True)

		if not os.path.exists(os.path.join(root, 'food101.tar.gz')):
			url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
			urllib.request.urlretrieve(url, os.path.join(root, 'food101.tar.gz'))

		if not os.path.exists(os.path.join(root, 'images')):
			os.system('tar -xzf {} --strip-components 1 -C {}'.format(os.path.join(root, 'food101.tar.gz'), root))

		# make directories
		os.makedirs(os.path.join(root, 'train'), exist_ok=True)
		os.makedirs(os.path.join(root, 'val'), exist_ok=True)
		for d in os.listdir(os.path.join(root, 'images')):
			if not d.startswith('._'):
				os.makedirs(os.path.join(root, 'train', d), exist_ok=True)
				os.makedirs(os.path.join(root, 'val', d), exist_ok=True)
	
		with open(os.path.join(root, 'meta/train.txt'), 'r') as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip() # remove new line
				if os.path.exists(os.path.join(root, 'images/{}.jpg'.format(line))):
					shutil.move(os.path.join(root, 'images/{}.jpg'.format(line)), os.path.join(root, 'train/{}.jpg'.format(line)))

		with open(os.path.join(root, 'meta/test.txt'), 'r') as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip() # remove new line
				if os.path.exists(os.path.join(root, 'images/{}.jpg'.format(line))):
					shutil.move(os.path.join(root, 'images/{}.jpg'.format(line)), os.path.join(root, 'val/{}.jpg'.format(line)))

		shutil.rmtree(os.path.join(root, 'images'))

	# === flower102 === #
	elif name == 'flower102':
		os.makedirs(root, exist_ok=True)

		if not os.path.exists(os.path.join(root, 'flower102.tgz')):
			url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/flower102.tgz'
			urllib.request.urlretrieve(url, os.path.join(root, 'flower102.tgz'))

		if not os.path.exists(os.path.join(root, 'imagelabels.mat')):
			url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
			urllib.request.urlretrieve(url, os.path.join(root, 'imagelables.mat'))

		if not os.path.exists(os.path.join(root, 'setid.mat')):
			url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'
			urllib.request.urlretrieve(url, os.path.join(root, 'setid.mat'))

		if not os.path.exists(os.path.join(root, 'jpg')):
			os.system('tar -xf {} -C {}'.format(os.path.join(root, 'flower102.tgz'), root))

		os.makedirs(os.path.join(root, 'train'), exist_ok=True)
		os.makedirs(os.path.join(root, 'val'), exist_ok=True)
		for i in range(102):
			os.makedirs(os.path.join(root, 'train/{:03d}'.format(i)), exist_ok=True)
			os.makedirs(os.path.join(root, 'val/{:03d}'.format(i)), exist_ok=True)

		setid_mat = io.loadmat(os.path.join(root, 'setid.mat'), squeeze_me=True)
		labels_mat = io.loadmat(os.path.join(root, 'imagelabels.mat'), squeeze_me=True)

		for id in setid_mat['trnid']:
			label = labels_mat['labels'][id - 1] - 1
			shutil.move(os.path.join(root, 'jpg/image_{:05d}.jpg'.format(id)), os.path.join(root, 'train/{:03d}/image_{:05d}.jpg'.format(label, id)))

		for id in setid_mat['tstid']:
			label = labels_mat['labels'][id - 1] - 1
			shutil.move(os.path.join(root, 'jpg/image_{:05d}.jpg'.format(id)), os.path.join(root, 'val/{:03d}/image_{:05d}.jpg'.format(label, id)))



class CustomTensorDataset(Dataset):
	"""Custom Tensor Dataset"""
	def __init__(self, root, transform=None, target_transform=None):
		"""
		Args:
		- root (str): root directory
		- transform (callable, optiional): transform function for data
		- target_transform (callable, optional): transform function for target label
		"""
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.data, self.labels = torch.load(self.root)

	def __getitem__(self, index):
		"""
		Args:
		- index (int): index
		"""
		img, target = self.data[index], self.labels[index]

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)
		
		return img, target

	def __len__(self):
		return len(self.data)
