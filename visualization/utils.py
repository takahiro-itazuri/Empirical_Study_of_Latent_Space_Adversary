import os
import sys

import torch

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *


__all__ = [
	'normalize_and_adjust',
	'clamp',
	'add_noise'
]


def normalize_and_adjust(x, name, ratio=3.0, device='cpu'):
	"""
	Normalize and adjust ``x`` for visualization 

	Args:
	x (torch.Tensor): input value (B, C, H, W)
	name (str): dataset name
	device (str): device
	"""
	# normalize
	std = x.std().item()
	x = torch.clamp(x, min=-ratio*std, max=ratio*std) / (ratio*std)

	# adjust
	mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(device)
	std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(device)

	x.mul_(std[None, :, None, None]).add_(mean[None, :, None, None])
	return x


def clamp(x, name, device='cpu'):
	"""
	Returns clamped ``x`` based on dataset stats

	Args:
	x (torch.Tensor): input value (B, C, H, W)
	name (str): dataset name
	device (str): device
	"""

	return normalize(unnormalize(x, name, device).clamp(min=0.0, max=1.0), name, device)


def add_noise(x, std, name, device='cpu'):
	"""
	Returns ``x`` + noise ([-std, std])

	Args:
	x (toch.Tensor): input value
	name (str): dataset name
	device (str): device
	"""

	n = torch.randn(x.shape).to(device) * std * 2.0
	return clamp(x + n, name, device=device)

