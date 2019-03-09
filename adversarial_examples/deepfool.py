import os
import sys
import copy
import numpy as np

import torch
torch.backends.cudnn.benchmark=True
import torchvision
from torch.autograd.gradcheck import zero_gradients
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import grad
from torchvision import transforms
from torchvision.utils import save_image

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *
from adversarial_examples.options import DeepFoolOptions
from adversarial_examples.utils import *


def deepfool_attack(x, t, model, dataset, num_candidates=-1, overshoot=0.2, max_itr=10, device='cpu'):
	"""
	Args:
	- x (torch.Tensor): input image
	- t (torch.Tensor): target label
	- classifier (nn.Module): target classifier
	- dataset (str): dataset name
	- num_candidates (int): number of candidate classes to attack
	- overshoot (float): overshoot value (eta in the original paper)
	- max_itr (int): max iterations
	- device (str): device
	"""

	if num_candidates == -1:
		num_candidates = len(get_labels(dataset)) - 1
		
	candidates = model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)).flatten().argsort(descending=True)[1:num_candidates+1]
	k_x0 = t.item()

	xi = copy.deepcopy(x).to(device)
	xi.requires_grad_()
	zero_gradients(xi)
	f_xi = model(xi if xi.shape[1] == 3 else xi.repeat(1, 3, 1, 1))
	k_xi = f_xi.argmax().item()
	
	itr = 0
	while (k_xi == k_x0) and (itr < max_itr):
		norm_l = float('inf')
		df_xi_kx0 = grad(f_xi[0][k_x0], xi, retain_graph=True)[0].detach()

		for j in range(num_candidates):
			zero_gradients(xi)
			k_xi = candidates[j]
			df_xi_kxi = grad(f_xi[0][k_xi], xi, retain_graph=True)[0].detach()

			w_k = df_xi_kxi - df_xi_kx0
			f_k = (f_xi[0][k_xi] - f_xi[0][k_x0]).detach()

			norm_k = torch.abs(f_k).item() / w_k.norm().item()

			if norm_k < norm_l:
				norm_l = norm_k
				w_l = w_k

		ri = ((norm_l + 1e-6) * w_l) / w_l.norm()
		xi = clamp(xi + (1.0 + overshoot) * ri, dataset, device)
		itr += 1

		zero_gradients(xi)
		f_xi = model(xi if xi.shape[1] == 3 else xi.repeat(1, 3, 1, 1))
		k_xi = f_xi.argmax().item()
	
	return xi


def main():
	opt = DeepFoolOptions().parse()

	# dataset
	dataset = shuffle_dataset(get_dataset(opt.dataset, train=opt.use_train, input_size=opt.input_size, augment=False))
	loader = DataLoader(dataset, batch_size=1, shuffle=False)
	labels = get_labels(opt.dataset)
	opt.num_classes = len(labels)
	
	# model
	model = get_classifier(opt.arch, opt.num_classes, opt.pretrained).to(opt.device)
	if opt.weight != None:
		model.load_state_dict(torch.load(opt.weight))
	model.eval()

	# make directories
	for i in range(opt.num_classes):
		for j in range(opt.num_classes):
			os.makedirs(os.path.join(opt.log_dir, '{:03d}/{:03d}'.format(i, j)), exist_ok=True)

	cnt = 0
	total = 0
	ntr = []
	aes = []
	labels = []

	for itr, (x, t) in enumerate(loader):
		x, t = x.to(opt.device), t.to(opt.device)

		init_pred = model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)).argmax()

		if init_pred.item() != t.item():
			continue

		perturbed_x = deepfool_attack(x, t, model, opt.dataset, opt.num_candidates, opt.overshoot, opt.max_itr, opt.device)

		final_pred = model(perturbed_x if perturbed_x.shape[1] == 3 else perturbed_x.repeat(1, 3, 1 ,1)).argmax()

		total += 1
		if final_pred.item() != t.item(): # attack succeeded
			cnt += 1

			ntr.append(transforms.functional.to_pil_image(unnormalize(x.cpu(), opt.dataset)[0]))
			aes.append(transforms.functional.to_pil_image(unnormalize(perturbed_x.cpu(), opt.dataset)[0]))
			labels.append(t.cpu())

			save_image(
				torch.cat((unnormalize(x.cpu(), opt.dataset), unnormalize(perturbed_x.cpu(), opt.dataset)), dim=0),
				os.path.join(opt.log_dir, '{:03d}/{:03d}/{:05d}.png'.format(init_pred.item(), final_pred.item(), itr)),
				padding=0
			)

			if opt.num_samples != -1:
				if cnt >= opt.num_samples:
					break
		
		if cnt % 10 == 0:
			sys.stdout.write('\r\033[Kprogress [{: 5d}/{: 5d}] {:d} adversarial examples are found from {:d} samples.'.format(itr+1, len(dataset), cnt, total))
			sys.stdout.flush()

	# save as one file
	labels = torch.tensor(labels)
	with open(os.path.join(opt.log_dir, 'ntr.pt'), 'wb') as f:
		torch.save((ntr, labels), f, pickle_protocol=4)
	with open(os.path.join(opt.log_dir, 'aes.pt'), 'wb') as f:
		torch.save((aes, labels), f, pickle_protocol=4)

	sys.stdout.write('\r\033[K{:d} adversarial examples are found from {:d} samples.\n'.format(cnt, total))
	sys.stdout.write('success rate {:.2f}\n'.format(float(cnt)/float(total)))
	sys.stdout.flush()


if __name__ == '__main__':
	main()