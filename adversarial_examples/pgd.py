import os
import sys
import numpy as np
from PIL import Image

import torch
torch.backends.cudnn.benchmark=True
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import grad
from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms
from torchvision.utils import save_image

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *
from adversarial_examples.options import PGDOptions
from adversarial_examples.utils import *


def pgd_attack(x, t, model, dataset, eps, alpha=2.0/255.0, num_steps=5, restarts=1, randomize=True, p=-1, device="cpu"):
	"""
	Args:
	- x (torch.Tensor): input image
	- t (int): target label
	- model (nn.Module): target classifier
	- dataset (str): dataset name
	- eps (float): max size of total pertubation 
	- alpha (float): perturbation size of single step
	- num_steps (int): number of steps
	- restarts (int): times of restarting
	- randomize (bool): if True, applay random initialization 
	- p (int): l-p norm (-1 means infty norm)
	- dataset (str): dataset name
	- device (str): device 
	"""
	assert eps >= 0.0
	assert alpha >= 0.0
	assert num_steps >= 1
	assert p >= -1
	if not randomize:
		restarts = 1
	
	eps = scale(eps, dataset)[None, :, None, None].to(device)
	alpha = scale(alpha, dataset)[None, :, None, None].to(device)

	max_loss  = -float('inf')
	max_delta = torch.zeros_like(x) 

	for i in range(restarts):
		# randomization
		if randomize:
			delta = torch.rand_like(x, requires_grad=True)
			delta.data = (2.0 * delta.data - 1.0) * eps
		else:
			delta = torch.zeros_like(x, requires_grad=True)

		for j in range(num_steps):
			zero_gradients(delta)
			perturbed_x = x + delta
			loss = F.cross_entropy(model(perturbed_x if perturbed_x.shape[1] == 3 else perturbed_x.repeat(1, 3, 1, 1)), t, reduction='sum')
			grad_delta = grad(loss, delta)[0].detach()

			# normalization
			if p == -1:
				delta.data = clamp_minmax(delta.data + alpha * grad_delta.sign(), min=-eps, max=eps)
				delta.data = clamp(x + delta.data, dataset, device) - x
			elif p >= +1:
				grad_delta_norm = calc_norm_each_sample(grad_delta)
				delta.data = delta.data + alpha * grad_delta / grad_delta_norm
				delta.data = clamp(x * delta.data, dataset, device) - x

				delta_norm = calc_norm_each_sample(delta.data)
				delta.data *= eps / clamp_min(delta_norm, min=eps.mean(dim=1, keepdim=True))
			else:
				raise NotImplementedError

		perturbed_x = x + delta
		final_loss = F.cross_entropy(model(perturbed_x if perturbed_x.shape[1] == 3 else perturbed_x.repeat(1, 3, 1, 1)), t).cpu().item()
		if final_loss > max_loss:
			max_loss  = final_loss
			max_delta = delta.detach()

	return clamp(x + max_delta.data, dataset, device)


def main():
	opt = PGDOptions().parse()

	# dataset
	dataset = shuffle_dataset(get_dataset(opt.dataset, train=opt.use_train, input_size=opt.input_size))
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
	ts = []

	for itr, (x, t) in enumerate(loader):
		x, t = x.to(opt.device), t.to(opt.device)

		init_pred = model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)).argmax()

		if init_pred.item() != t.item():
			continue

		perturbed_x = pgd_attack(x, t, model, opt.dataset, opt.eps, opt.alpha, opt.num_steps, opt.restarts, True, opt.p, opt.device)

		final_pred = model(perturbed_x if perturbed_x.shape[1] == 3 else perturbed_x.repeat(1, 3, 1 ,1)).argmax()

		total += 1
		if final_pred.item() != t.item(): # attack succeeded
			cnt += 1

			ntr.append(transforms.functional.to_pil_image(unnormalize(x.cpu(), opt.dataset)[0]))
			aes.append(transforms.functional.to_pil_image(unnormalize(perturbed_x.cpu(), opt.dataset)[0]))
			ts.append(t.cpu())

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
	ts = torch.tensor(ts)
	with open(os.path.join(opt.log_dir, 'ntr.pt'), 'wb') as f:
		torch.save((ntr, ts), f, pickle_protocol=4)
	with open(os.path.join(opt.log_dir, 'aes.pt'), 'wb') as f:
		torch.save((aes, ts), f, pickle_protocol=4)

	sys.stdout.write('\r\033[K{:d} adversarial examples are found from {:d} samples.\n'.format(cnt, total))
	sys.stdout.write('success rate {:.2f}\n'.format(float(cnt)/float(total)))
	sys.stdout.flush()

	save_result({
		'cnt': cnt,
		'total': total,
		'robust accuracy': 100.0 * (1.0 - float(cnt) / float(total)),
		'success rate': 100.0 * float(cnt) / float(total)
	}, opt.log_dir, opt.result)


if __name__ == '__main__':
	main()
