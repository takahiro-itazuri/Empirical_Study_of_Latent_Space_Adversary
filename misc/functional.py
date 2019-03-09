import os
import sys

import torch


__all__ = [
	'label2onehot',
	'calc_norm_each_sample',
	'truncated_normal'
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


def truncated_normal(shape, mean=0.0, std=1.0):
	x = torch.randn(shape)
	tmp = x.new_empty(shape + (4,)).normal_()
	valid = (tmp < 2) & (tmp > -2)
	idx = valid.max(-1, keepdim=True)[1]
	x.data.copy_(tmp.gather(-1, idx).squeeze(-1))
	x.data.mul_(std).add_(mean)
	return x
