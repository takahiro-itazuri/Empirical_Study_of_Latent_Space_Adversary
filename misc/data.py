import os
import sys
import csv
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
	'mnist'         : [0.13066049],
	'svhn'          : [0.43768210, 0.44376970, 0.47280442],
	'cifar10'       : [0.49139968, 0.48215841, 0.44653091],
	'cifar100'      : [0.50707516, 0.48654887, 0.44091784],
	'stl10'         : [0.44671062, 0.43980984, 0.40664645],
	'lsun'          : [0.485, 0.456, 0.406], # copied from imagenet
	'cub200'        : [0.48310599, 0.49175689, 0.42481980],
	'dog120'        : [0.47809049, 0.44543245, 0.38800871],
	'food101'       : [0.55774122, 0.44237375, 0.32717270],
	'flower102'     : [0.51136059, 0.41595200, 0.34071067],
	'tiny_imagenet' : [0.48234981, 0.44617516, 0.39390695],
	'imagenet'      : [0.485, 0.456, 0.406]
}

stds = {
	'mnist'         : [0.30810780],
	'svhn'          : [0.19803012, 0.20101562, 0.19703614],
	'cifar10'       : [0.24703223, 0.24348513, 0.26158784],
	'cifar100'      : [0.26733429, 0.25643846, 0.27615047],
	'stl10'         : [0.26034098, 0.25657727, 0.27126738],
	'lsun'          : [0.229, 0.224, 0.225], # copied from imagenet
	'cub200'        : [0.22814971, 0.22405523, 0.25914747],
	'dog120'        : [0.25985241, 0.25299320, 0.25506458],
	'food101'       : [0.25912008, 0.26311737, 0.26589110],
	'flower102'     : [0.29568306, 0.24932568, 0.28895128],
	'tiny_imagenet' : [0.26266909, 0.25351605, 0.26663900],
	'imagenet'      : [0.229, 0.224, 0.225]
}

dataset_list = ['mnist', 'svhn', 'cifar10', 'cifar100', 'stl10', 'lsun', 'cub200', 'dog120', 'food101', 'flower102', 'tiny_imagenet', 'imagenet']


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
	dataset = ImageFolder(
		path, 
		transform=transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224), 
			transforms.ToTensor()
		])
	)
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
	elif name in ['lsun', 'cub200', 'dog120', 'food101', 'flower102', 'tiny_imagenet', 'imagenet']:
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

		if not os.path.exists(os.path.join(root, 'CUB_200_2011.tgz')):
			url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
			urllib.request.urlretrieve(url, os.path.join(root, 'CUB_200_2011.tgz'))

		if not os.path.exists(os.path.join(root, 'images')):
			os.system('tar -xzf {} --strip-components 1 -C {}'.format(os.path.join(root, 'CUB_200_2011.tgz'), root))

		# make directories
		os.makedirs(os.path.join(root, 'train'), exist_ok=True)
		os.makedirs(os.path.join(root, 'val'), exist_ok=True)
		for d in os.listdir(os.path.join(root, 'images')):
			if not d.startswith('._'):
				os.makedirs(os.path.join(root, 'train', d), exist_ok=True)
				os.makedirs(os.path.join(root, 'val', d), exist_ok=True)

		images = []
		with open(os.path.join(root, 'images.txt'), 'r') as f:
			lines = f.readlines()
			for line in lines:
				images.append(line.strip().split()[1])

		with open(os.path.join(root, 'train_test_split.txt'), 'r') as f:
			lines = f.readlines()
			for i, line in enumerate(lines):
				label = line.strip().split()[1]
				print(images[i], label)
				if label == '0':
					shutil.move(os.path.join(root, 'images', images[i]), os.path.join(root, 'val', images[i]))
				elif label == '1':
					shutil.move(os.path.join(root, 'images', images[i]), os.path.join(root, 'train', images[i]))

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
	def is_within_directory(directory, target):
		
		abs_directory = os.path.abspath(directory)
		abs_target = os.path.abspath(target)
	
		prefix = os.path.commonprefix([abs_directory, abs_target])
		
		return prefix == abs_directory
	
	def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
	
		for member in tar.getmembers():
			member_path = os.path.join(path, member.name)
			if not is_within_directory(path, member_path):
				raise Exception("Attempted Path Traversal in Tar File")
	
		tar.extractall(path, members, numeric_owner=numeric_owner) 
		
	
	safe_extract(tf, os.path.join(root))

		if not os.path.exists(os.path.join(root, 'lists')):
			with tarfile.open(os.path.join(root, 'lists.tar'), 'r') as tf:
	def is_within_directory(directory, target):
		
		abs_directory = os.path.abspath(directory)
		abs_target = os.path.abspath(target)
	
		prefix = os.path.commonprefix([abs_directory, abs_target])
		
		return prefix == abs_directory
	
	def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
	
		for member in tar.getmembers():
			member_path = os.path.join(path, member.name)
			if not is_within_directory(path, member_path):
				raise Exception("Attempted Path Traversal in Tar File")
	
		tar.extractall(path, members, numeric_owner=numeric_owner) 
		
	
	safe_extract(tf, os.path.join(root,"lists"))

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

	# === Food-101 === #
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

	# === Flower-102 === #
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

	# === tiny ImageNet === #
	elif name == 'tiny_imagenet':
		os.makedirs(root, exist_ok=True)

		if not os.path.exists(os.path.join(root, 'tiny-imagenet-200.zip')):
			url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
			urllib.request.urlretrieve(url, os.path.join(root, 'tiny-imagenet-200.zip'))

		# rename root directories
		if not os.path.exists(os.path.join(root, 'train')):
			with zipfile.ZipFile(os.path.join(root, 'tiny-imagenet-200.zip')) as z:
				z.extractall(root)
			shutil.move(os.path.join(root, 'tiny-imagenet-200/train'), os.path.join(root, 'train'))
			shutil.move(os.path.join(root, 'tiny-imagenet-200/val'), os.path.join(root, 'val'))
			shutil.move(os.path.join(root, 'tiny-imagenet-200/test'), os.path.join(root, 'test'))

		# move train images
		for d in os.listdir(os.path.join(root, 'train')):
			os.system('mv {} {}'.format(
				os.path.join(root, 'train', d, 'images/*'),
				os.path.join(root, 'train', d)
			))
			os.rmdir(os.path.join(root, 'train', d, 'images'))
			os.remove(os.path.join(root, 'train', d, '{}_boxes.txt'.format(d)))

		# move val images
		with open(os.path.join(root, 'val/val_annotations.txt'), 'r') as f:
			reader = csv.reader(f, delimiter='\t')
			for line in reader:
				os.makedirs(os.path.join(root, 'val', line[1]), exist_ok=True)
				shutil.move(
					os.path.join(root, 'val/images', line[0]),
					os.path.join(root, 'val', line[1])
				)
		
		os.rmdir(os.path.join(root, 'val/images'))
		os.remove(os.path.join(root, 'val/val_annotations.txt'))

		shutil.rmtree(os.path.join(root, 'tiny-imagenet-200'))


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
