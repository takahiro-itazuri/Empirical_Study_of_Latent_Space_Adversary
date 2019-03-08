import os
import sys
import random

import torch
import torchvision
from torch import nn
from torch.autograd import grad, Variable
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import Subset, DataLoader
from torchvision.utils import save_image

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *
from visualization.options import GradientsOptions
from visualization.utils import normalize_and_adjust, add_noise


class VanillaGrad():
	def __init__(self, model, num_classes, device):
		self.model = model
		self.model.eval()
		self.num_classes = num_classes
		self.device = device

	def __call__(self, x, t):
		x.requires_grad = True
		y = self.model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1))

		self.model.zero_grad()
		one_hot_label = label2onehot(t.item(), self.num_classes).to(self.device)

		y.backward(gradient=one_hot_label)
		grad = x.grad.data

		return grad


class LossGrad():
	def __init__(self, model, num_classes, device, criterion=nn.CrossEntropyLoss()):
		self.model = model
		self.model.eval()
		self.num_classes = num_classes
		self.device = device
		self.criterion = criterion

	def __call__(self, x, t):
		x.requires_grad = True
		y = self.model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1))
		loss = self.criterion(y, t)

		zero_gradients(x)
		self.model.zero_grad()
		loss.backward(retain_graph=True)
		grad = x.grad.data

		return grad


class SmoothGrad(VanillaGrad):
	def __init__(self, model, num_classes, dataset, device, normalization=0.2, num_samples=50):
		super(SmoothGrad, self).__init__(model, num_classes, device)
		self.dataset = dataset
		self.normalization = normalization
		self.num_samples = num_samples
  
	def __call__(self, x, t):
		std = self.normalization * (x.max() - x.min())
		smooth_gradient = torch.zeros_like(x).to(self.device)
		x.requires_grad = True

		noisy_x = []
		for i in range(self.num_samples):
			noisy_x.append(add_noise(x, std / 2.0, self.dataset, self.device))
		noisy_x = torch.cat(noisy_x, dim=0).to(self.device)
		noisy_x = Variable(noisy_x, requires_grad=True)

		y = self.model(noisy_x if noisy_x.shape[1] == 3 else noisy_x.repeat(1, 3, 1, 1))

		one_hot_label = label2onehot(t.item(), self.num_classes).to(self.device)
		one_hot_label = one_hot_label.repeat(self.num_samples, 1)

		y.backward(gradient=one_hot_label, retain_graph=True)
		grad = torch.mean(noisy_x.grad.data, dim=0, keepdim=True)

		return grad


class GuidedBackprop():
	def __init__(self, model, num_classes, device):
		self.model = model
		self.model.eval()
		self.num_classes = num_classes
		self.device = device

		self._hook_relu()

	def _hook_relu(self):
		def _relu_hook_func(module, grad_input, grad_output):
			if isinstance(module, nn.ReLU):
				return (torch.clamp(grad_input[0], min=0.0),)
      
		for module in self.model.modules():
			module.register_backward_hook(_relu_hook_func)
  
	def __call__(self, x, t):
		x.requires_grad = True
		y = self.model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1))

		self.model.zero_grad()
		one_hot_label = label2onehot(t.item(), self.num_classes).to(self.device)

		y.backward(gradient=one_hot_label)
		grad = x.grad.data

		return grad


def main():
	opt = GradientsOptions().parse()

	dataset = get_dataset(opt.dataset, opt.use_train)
	if opt.num_samples != -1:
		opt.num_samples = min(opt.num_samples, len(dataset))
		random.seed(0)
		indices = range(len(dataset))
		indices = random.sample(indices, opt.num_samples)
		dataset = Subset(dataset, indices)
	loader = DataLoader(dataset, batch_size=1, shuffle=False)
	labels = get_labels(opt.dataset)
	opt.num_classes = len(labels)

	model = get_classifier(opt.arch, opt.num_classes, opt.pretrained).to(opt.device)
	if opt.weight != None:
		model.load_state_dict(torch.load(opt.weight))
	model.eval()

	if opt.method == 'vanilla':
		visualizer = VanillaGrad(model, opt.num_classes, opt.device)
	elif opt.method == 'loss':
		visualizer = LossGrad(model, opt.num_classes, opt.device)
	elif opt.method == 'smooth':
		visualizer = SmoothGrad(model, opt.num_classes, opt.dataset, opt.device)
	elif opt.method == 'guided':
		visualizer = GuidedBackprop(model, opt.num_classes, opt.device)
	else:
		raise NotImplementedError

	for i in range(opt.num_classes):
		os.makedirs(os.path.join(opt.log_dir, '{:03d}'.format(i)), exist_ok=True)

	for itr, (x, t) in enumerate(loader):
		x, t = x.to(opt.device), t.to(opt.device)

		grad = visualizer(x, t)

		img = torch.cat((unnormalize(x.detach(), opt.dataset, opt.device), normalize_and_adjust(grad.detach(), opt.dataset, opt.ratio, opt.device)), dim=0)

		save_image(
			img,
			os.path.join(opt.log_dir, '{:03d}/{:03d}.png'.format(t.cpu().item(), itr)),
			padding=0
		)

		itr += 1
		if itr % 100 == 0:
			sys.stdout.write('\r\033[Kitr [{:d}/{:d}]'.format(itr, len(dataset)))
			sys.stdout.flush()

	sys.stdout.write('\r\033[Kitr [{:d}/{:d}]\n'.format(itr, len(dataset)))
	sys.stdout.flush()


if __name__ == '__main__':
	main()