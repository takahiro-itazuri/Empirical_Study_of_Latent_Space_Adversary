"""
This code is the implementation of conditional batch normalization.
This is implemented with reference to following links.
- https://github.com/pytorch/pytorch/issues/8985#issuecomment-405071175
- https://discuss.pytorch.org/t/conditional-batch-normalization/14412
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalBatchNorm2d(nn.Module):
	def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
		super(ConditionalBatchNorm2d, self).__init__()
		self.num_features = num_features
		self.num_classes = num_classes
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		if self.affine:
			self.weight = nn.Parameter(torch.Tensor(num_classes, num_features))
			self.bias = nn.Parameter(torch.Tensor(num_classes, num_features))
		else:
			self.register_parameter('weight', None)
			self.register_parameter('bias', None)
		if self.track_running_stats:
			self.register_buffer('running_mean', torch.zeros(num_features))
			self.register_buffer('running_var', torch.ones(num_features))
			self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
		else:
			self.register_parameter('running_mean', None)
			self.register_parameter('running_var', None)
			self.register_parameter('num_batches_tracked', None)
		self.reset_paraemters()

	def reset_runnging_stats(self):
		if self.track_running_stats:
			self.running_mean.zero_()
			self.running_var.fill_(1)
			self.num_batches_tracked.zero_()

	def reset_paraemters(self):
		self.reset_runnging_stats()
		if self.affine:
			self.weight.data.uniform_()
			self.bias.data.zero_()

	def _check_input_dim(self, input):
		if input.dim() != 4:
			raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

	def forward(self, input, label):
		self._check_input_dim(input)

		exponential_average_factor = 0.0

		if self.training and self.track_running_stats:
			self.num_batches_tracked += 1
			if self.momentum is None:
				exponential_average_factor = 1.0 / self.num_batches_tracked.item()
			else:
				exponential_average_factor = self.momentum

		out = F.batch_norm(input, self.running_mean, self.running_var, None, None, self.training or not self.track_running_stats, exponential_average_factor, self.eps)
		if self.affine:
			shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
			weight = self.weight.index_select(0, label.long()).view(shape)
			bias = self.bias.index_select(0, label.long()).view(shape)
			out = out * weight + bias
		return out

	def __repr__(self):
		return ('{name}({num_features}, eps={eps}, momentum={momentum})'.format(name=self.__class__.__name__, **self.__dict__))
