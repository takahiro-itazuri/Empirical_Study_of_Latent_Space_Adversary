import torch
from torch import nn


class LayerNorm(nn.Module):
	def __init__(self, num_features, eps=1e-5, affine=True):
		super(LayerNorm, self).__init__()
		self.eps = eps
		self.affine = affine

		if self.affine:
			self.gamma = nn.Parameter(torch.ones(num_features))
			self.beta = nn.Parameter(torch.zeros(num_features))

	def forward(self, x):
		shape = [-1] + [1] * (x.dim() - 1)
		mean = x.view(x.size(0), -1).mean(1).view(*shape)
		std = x.view(x.size(0), -1).std(1).view(*shape)
		x = (x - mean) / (std + self.eps)

		if self.affine:
			shape = [1, -1] + [1] * (x.dim() - 2)
			x = self.gamma.view(*shape) * x + self.beta.view(*shape)

		return x
