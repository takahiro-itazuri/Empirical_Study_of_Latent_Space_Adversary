import os
import sys
import copy

import torch
torch.backends.cudnn.benchmark=True
from torch import nn, optim
from torch.autograd import grad, Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision import models
from torchvision.utils import save_image

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *
from visualization.options import ActivationMaximizationOptions
from visualization.utils import normalize_and_adjust


def main():
	opt = ActivationMaximizationOptions().parse()

	opt.labels = get_labels(opt.dataset)
	opt.num_classes = len(opt.labels)
	opt.channel = 1 if opt.dataset == 'mnist' else 3

	# model
	model = get_classifier(opt.arch, num_classes=opt.num_classes, inplace=False, pretrained=opt.pretrained).to(opt.device)
	if opt.weight != None:
		model.load_state_dict(torch.load(opt.weight))
	model.eval()

	# hook
	conv_outputs = []
	def hook_func(module, input, output):
		conv_outputs.append(output[:, :opt.num_filters])

	for idx, module in enumerate(model.modules()):
		if isinstance(module, nn.Conv2d):
			module.register_forward_hook(hook_func)

	# get size
	input = torch.zeros(1, opt.channel, 224, 224).to(opt.device)
	model(input if input.shape[1] == 3 else input.repeat(1, 3, 1, 1))
	size = len(conv_outputs) * opt.num_filters

	# initialize
	inputs = []
	optimizers = []
	for n in range(size):
		inputs.append(Variable(opt.eps * (2.0 * torch.rand(input.shape).to(opt.device) - 1.0), requires_grad=True))
		optimizers.append(optim.Adam([inputs[n]], lr=opt.lr, weight_decay=opt.wd))

	# optimize
	for itr in range(opt.num_itrs):
		if (itr+1) % 10 == 0:
			sys.stdout.write('\r\033[K[itr {:d}] processing...'.format(itr+1))
			sys.stdout.flush()

		for n in range(size):
			conv_outputs = []
			optimizers[n].zero_grad()
			model(inputs[n] if inputs[n].shape[1] == 3 else inputs[n].repeat(1, 3, 1, 1))

			b = n // opt.num_filters
			l = n - b * opt.num_filters
			conv_output = conv_outputs[b][0][l] # (block, batch=1, layer)
			loss = -torch.mean(conv_output)
			loss.backward()
			optimizers[n].step()

	sys.stdout.write('\r\033[K[itr {:d}] done!\n'.format(opt.num_itrs))
	sys.stdout.flush()

	for n in range(size):
		filter = normalize_and_adjust(inputs[n].detach(), opt.dataset, opt.ratio, opt.device)
		b = n // opt.num_filters
		l = n - b * opt.num_filters
		save_image(filter, os.path.join(opt.log_dir, 'conv{:02d}-{:02d}.png'.format(b, l)))


if __name__ == '__main__':
	main()
