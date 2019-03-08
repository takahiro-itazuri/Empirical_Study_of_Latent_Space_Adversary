import os
import sys

import torch


__all__ = [
	'label2onehot',
	'calc_norm_each_sample'
]


def label2onehot(label, num_classes, dtype=torch.float32):
	one_hot_label = torch.zeros((1, num_classes), dtype=dtype)
	one_hot_label[0][label] = 1
	return one_hot_label


def calc_norm_each_sample(x, p=2):
	"""
	Calculate Lp norm for each sample

	Args:
	x (torch.Tensor): input value (B, C, H, W)
	p (int): order of norm

	Returns:
	norm (torch.Tensor): norm (B, 1, 1, 1)
	"""
	return x.view(x.size(0), -1).norm(p=p, dim=1)[:, None, None, None]

