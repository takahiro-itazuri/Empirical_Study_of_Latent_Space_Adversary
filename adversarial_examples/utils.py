import os
import sys

import torch

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from misc import *


__all__ = [
	'get_eps',
	'scale',
	'clamp',
	'clamp_minmax',
	'clamp_max',
	'clamp_min'
]


def Linf2L2(ep_Linf, dim):
	# ep_L2 = sqrt(2*dim / pi*e) * ep_Linf
	assert type(dim)==int
	return 0.48 * sqrt(dim) *ep_Linf 


def get_eps(mode, p, dataset):
	"""
	Get eps value from the order of norm and dataset name

	Args:
	mode (str): mode
	p (int): order of norm
	dataset (str): dataset name
	"""
	eps = 0.0

	if mode == 'train':
		if dataset == 'mnist':
			if p == -1: eps = 0.3
			elif p == 2: eps = 1.5
			else:
				raise NotImplementedError
		elif dataset == 'cifar10':
			if p == -1: eps = 0.0157
			elif p == 2: eps = 0.314
			else:
				raise NotImplementedError
		elif dataset == 'svhn':
			if p == -1: eps = 0.0157
			elif p == 2: eps = 0.314
			else:
				raise NotImplementedError
		elif dataset == 'lsun':
			if p == -1: eps = 0.0157
			elif p == 2: eps = 0.314
			else:
				raise NotImplementedError
		elif dataset == 'stl10':
			if p == -1: eps = 0.0157
			elif p == 2: eps = 0.314
			else:
				raise NotImplementedError
		elif dataset == 'imagenet':
			if p == -1: eps = 0.005
			elif p == 2: eps = 1.0
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError

	elif mode == 'large':
		if dataset == 'mnist':
			if p == -1: eps = 0.3
			elif p == 2: eps = 4
			else:
				raise NotImplementedError
		elif dataset == 'cifar10':
			if p == -1: eps = 0.125
			elif p == 2: eps = 4.7
			else:
				raise NotImplementedError
		elif dataset == 'svhn':
			if p == -1: eps = 0.125
			elif p == 2: eps = 4.7
			else:
				raise NotImplementedError
		elif dataset == 'lsun':
			if p == -1: eps = 0.125
			elif p == 2: eps = 4.7
			else:
				raise NotImplementedError
		elif dataset == 'stl10':
			if p == -1: eps = 0.125
			elif p == 2: eps = 4.7
			else:
				raise NotImplementedError
		elif dataset == 'imagenet':
			if p == -1: eps = 0.25
			elif p == 2: eps = 40
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError

	return eps


def get_alpha(eps, num_steps, ratio=1.5):
	"""
	Get alpha value given eps value and number of steps

	Args:
	eps (float): epsilon value
	num_steps (int): number of steps
	"""

	return ratio * eps / num_steps


def scale(x, name, device='cpu'):
	"""
	Returns scaled ``x`` based on dataset stats

	Args:
	x (float / list): input value
	name (str): dataset name
	device (str): device
	"""
	_, std = get_dataset_stats(name)
	std = std.to(device)

	if isinstance(x, list):
		return torch.tensor(x).expand_as(std) / std

	elif isinstance(x, float):
		return torch.tensor([x]).expand_as(std) / std

	else:
		raise NotImplementedError


def clamp(x, name, device='cpu'):
	"""
	Returns clamped ``x`` based on dataset stats

	Args:
	x (torch.Tensor): input value (B, C, H, W)
	name (str): dataset name
	device (str): device
	"""

	return normalize(unnormalize(x, name, device).clamp(min=0.0, max=1.0), name, device)


def clamp_minmax(x, min, max):
	"""
	Returns clamped ``x`` for each channel

	Args:
	x (torch.Tensor): input value (B, C, H, W)
	min (torch.Tensor): minimal value (1, C, 1, 1)
	max (torch.Tensor): maximum value (1, C, 1, 1)
	"""
	min = min.expand_as(x)
	max = max.expand_as(x)
	x = torch.where(x > min, x, min)
	return torch.where(x < max, x, max)


def clamp_min(x, min):
	"""
	Returns clamped ``x`` for each channel

	Args:
	x (torch.Tensor): input value (B, C, H, W)
	min (torch.Tensor): minimal value (1, C, 1, 1)
	"""
	min = min.expand_as(x)
	return torch.where(x > min, x, min)


def clamp_max(x, max):
	"""
	Returns clamped ``x`` for each channel

	Args:
	x (torch.Tensor): input value (B, C, H, W)
	max (torch.Tensor): maximum value (1, C, 1, 1)
	"""
	max = max.expand_as(x)
	return torch.where(x < max, x, max)
