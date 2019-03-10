"""
This code is the implementation of "Constructing Unrestricted Adversarial Examples with Generative Models".
(http://papers.nips.cc/paper/8052-constructing-unrestricted-adversarial-examples-with-generative-models)

The original implementation by TensorFlow is available.
(https://github.com/ermongroup/generative_adversary)
"""

import os
import sys
import copy
import random

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *
from models import *
from wgan_gp.utils import *
from latent_space_adversary.options import ConditionalLSAOptions
from latent_space_adversary.utils import *


def conditional_lsa_attack(z0, y_src, y_tgt, model, G, C, dataset, eps_z, lambda1, lambda2, alpha, max_itr, input_size, device):
	"""
	Args
	- z0 (torch.Tensor): initial latent variable
	- y_src (int): source label
	- y_tgt (int): target label
	- model (nn.Module): classifier
	- G (nn.Module): genrator of conditional WGAN-GP
	- C (nn.Module): critic of conditional WGAN-GP
	- dataset (str): dataset mame
	- eps_z (float): perturbation size
	- lambda1 (float): coefficient of loss1
	- lambda2 (float): coefficient of loss2
	- alpha (float): learning rate
	- max_itr (int): max iteration
	- input_size (int): input size of model
	- device (str): device nmmae
	"""
	zi = Variable(copy.deepcopy(z0), requires_grad=True).to(device)

	cls_criterion = nn.CrossEntropyLoss()
	ac_criterion = nn.CrossEntropyLoss()

	for itr in range(max_itr):
		xi = G(zi, y_src)
		xi_cls = gan2cls(xi, dataset, input_size, device)

		_, yi_ac = C(xi)
		xi = gan2cls(xi, dataset, input_size, device)
		yi_cls = model(xi if xi.shape[1] == 3 else xi.repeat(1, 3, 1, 1))

		if (yi_cls.argmax().item() == y_tgt.item()) and (yi_ac.argmax().item() == y_src.item()):
			return xi

		loss0 = cls_criterion(yi_cls, y_tgt)
		loss1 = torch.mean(torch.relu(torch.abs(zi - z0) - eps_z))
		loss2 = ac_criterion(yi_ac, y_src)
		loss = loss0 + lambda1 * loss1 + lambda2 * loss2
		loss.backward()

		zi = Variable(zi.detach() - alpha * zi.grad.data.detach(), requires_grad=True)

	return xi


def main():
	opt = ConditionalLSAOptions().parse()

	# dataset
	labels = get_labels(opt.dataset)
	opt.num_classes = len(labels)

	# model
	model = get_classifier(opt.arch, opt.num_classes).to(opt.device)
	model.load_state_dict(torch.load(opt.weight))
	model.eval()

	# wgan-gp
	G, C = get_wgan_gp(opt.nz, opt.nc, opt.nf, opt.gan_size, True, opt.num_classes)
	G, C = G.to(opt.device), C.to(opt.device)
	load_wgan_gp(G, C, opt.gan_dir)
	G.eval()
	C.eval()

	# make directories
	for i in range(opt.num_classes):
		for j in range(opt.num_classes):
			os.makedirs(os.path.join(opt.log_dir, '{:03d}/{:03d}'.format(i, j)), exist_ok=True)

	cnt = 0
	total = 0

	while True:
		if cnt >= opt.num_samples:
			break

		
		# generate label
		y_src = random.randrange(0, opt.num_classes)
		y_tgt = random.randrange(0, opt.num_classes)
		if y_tgt == y_src:
			y_tgt = (y_tgt + 1) % opt.num_classes
		y_src = torch.tensor([y_src]).to(opt.device)
		y_tgt = torch.tensor([y_tgt]).to(opt.device)
		
		# sample latent variable from truncated nromal distribution
		z0 = truncated_normal((1, opt.nz)).to(opt.device)
		x = gan2cls(G(z0, y_src).detach(), opt.dataset, opt.input_size, opt.device)

		init_pred = model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)).argmax()

		if init_pred.item() != y_src.item(): # classifier's prediction failed
			continue

		perturbed_x = conditional_lsa_attack(
			z0, y_src, y_tgt, model, G, C, opt.dataset, 
			opt.eps_z, opt.lambda1, opt.lambda2, opt.alpha, opt.max_itr, 
			opt.input_size, opt.device
		)

		final_pred = model(perturbed_x if perturbed_x.shape[1] == 3 else perturbed_x.repeat(1, 3, 1, 1)).argmax()
		
		total += 1
		if final_pred.item() == y_tgt.item(): # attack succeeded
			cnt += 1

			save_image(
				torch.cat((unnormalize(x.cpu(), opt.dataset), unnormalize(perturbed_x.cpu(), opt.dataset)), dim=0),
				os.path.join(opt.log_dir, '{:03d}/{:03d}/{:05d}.png'.format(y_src.item(), y_tgt.item(), cnt)),
				padding=0
			)

		if cnt % 10 == 0:
			sys.stdout.write('\r\033[K{:d} adversarial examples are found from {:d} samples.'.format(cnt, total))
			sys.stdout.flush()

	# save as one file
	ts = torch.tensor(ts)
	with open(os.path.join(opt.log_dir, 'ntr.pt'), 'wb') as f:
		torch.save((ntr, ts), f, pickle_protocol=4)
	with open(os.path.join(opt.log_dir, 'aes.pt'), 'wb') as f:
		torch.save((aes, ts), f, pickle_protocol=4)

	sys.stdout.write('\r\033[K{:d} adversarial examples are found from {:d} samples.\n'.format(cnt, total))
	sys.stdout.write('success rate {:.2f}\n'.format(float(cnt)/float(total)))
	sys.stdout.flush()


if __name__ == '__main__':
	main()
