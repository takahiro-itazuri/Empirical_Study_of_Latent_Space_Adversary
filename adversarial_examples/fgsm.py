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
from adversarial_examples.options import FGSMOptions
from adversarial_examples.utils import *


def fgsm_attack(x, t, model, dataset, eps, p=-1, device='cpu'):
	"""
	Args:
	- x (torch.Tensor): input image
	- t (torch.Tensor): target label
	- model (nn.Module): target classifier
	- dataset (str): dataset name
	- eps (float): perturbation size
	- p (int): order of norm (-1 means infty norm)
	- device (str): device
	"""
	eps = scale(eps, dataset)[None, :, None, None].to(device)
	delta = torch.zeros_like(x, requires_grad=True).to(device)

	# calculate gradient
	zero_gradients(delta)
	perturbed_x = x + delta
	loss = F.cross_entropy(model(perturbed_x if perturbed_x.shape[1] == 3 else perturbed_x.repeat(1, 3, 1, 1)), t, reduction='sum')
	grad_delta = grad(loss, delta)[0].detach()
	
	if p == -1:	# l_infty norm
		delta.data = eps * grad_delta.sign()

	elif p >= 1: #l_p norm
		grad_delta_norm = calc_norm_each_sample(grad_delta)
		delta.data = eps * grad_delta / grad_delta_norm
		
	else:
		raise NotImplementedError

	return clamp(x + delta, dataset, device)


def main():
	opt = FGSMOptions().parse()

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

		perturbed_x = fgsm_attack(x, t, model, opt.dataset, opt.eps, opt.p, opt.device)

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
	sys.stdout.write('\nsuccess rate {:.2f}\n'.format(float(cnt)/float(total)))
	sys.stdout.flush()

	save_result({
		'cnt': cnt,
		'total': total,
		'robust accuracy': 100.0 * (1.0 - float(cnt) / float(total)),
		'success rate': 100.0 * float(cnt) / float(total)
	}, opt.log_dir, opt.result)


if __name__ == '__main__':
	main()
