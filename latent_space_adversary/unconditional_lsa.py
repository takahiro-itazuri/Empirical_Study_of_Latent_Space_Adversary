"""
This code is the implementation of "Generating Natural Adversarial Examples".
(https://openreview.net/pdf?id=H1BLjgZCb)

The original implemention by TensorFlow is available.
(https://github.com/zhengliz/natural-adversary)
"""

import os
import sys

import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *
from models import *
from wgan_gp_inverter.utils import *
from latent_space_adversary.options import UnconditionalLSAOptions
from latent_space_adversary.utils import *


def unconditional_lsa_attack(x, t, model, G, C, I, dataset, method, max_r, dr, max_itr, batch_size, N, input_size, gan_size, device):
	"""
	Args:
	- x (torch.Tensor): input image
	- t (int): label
	- model (nn.Module): classifier
	- G (nn.Module): genrator of WGAN-GP with inverter
	- C (nn.Module): critic of WGAN-GP with inverter
	- I (nn.Module): inverter of WGAN-GP with inverter
	- dataset (str): dataset name
	- method (str): method name
	- max_r (float): max perturbation size
	- dr (float): perturbation size for single step
	- max_itr (int): max iteration
	- batch_size (int): batch size
	- N (int): number of samples
	- input_size (int): image size for classifier
	- gan_size (int): image size for GAN
	- device (str): device name
	"""

	with torch.no_grad():
		x = cls2gan(x, dataset, gan_size, device) # (1, C, H, W)
		z = I(x).detach() # (1, nz)
		nz = z.size(1)
		x_hat = G(z).detach()
		x_hat = gan2cls(x_hat, dataset, input_size, device)
		y_hat = model(x_hat if x_hat.shape[1] == 3 else x_hat.repeat(1, 3, 1, 1)).argmax()

		if y_hat.item() != t:
			return None

		if method == 'iterative_stochastic':
			r = 0
			x_ast = []
			z_ast = []

			while len(x_ast) == 0:
				cnt = 0
				while cnt < N:
					cnt += batch_size

					z_hat = z + (torch.rand((batch_size, 1)).to(device) * dr + r) * F.normalize(torch.randn(batch_size, nz).to(device), dim=1)
					x_hat = gan2cls(G(z_hat), dataset, input_size, device)
					y_hat = model(x_hat if x_hat.shape[1] == 3 else x_hat.repeat(1, 3, 1, 1)).argmax(dim=1, keepdim=True)

					mask = (y_hat != t).view(-1)
					x_hat = x_hat[mask]
					z_hat = z_hat[mask]

					for i in range(z_hat.size(0)):
						x_ast.append(x_hat[i].detach())
						z_ast.append(z_hat[i].detach())
			
				if r + dr <= max_r:
					r += dr
				else:
					return None

			minval = -1
			minidx = -1
			for i in range(len(z_ast)):
				val = torch.norm(z_ast[0] - z)
				if minidx < 0 or val < minval:
					minidx = i
					minval = val

			return x_ast[minidx].unsqueeze(0)


		elif method == 'hybrid_shrinking':
			x_ast_opt = None

			# recursive search
			l = 0
			r = max_r

			while r - l > dr:
				x_ast = []
				z_ast = []

				cnt = 0
				while cnt < N:
					cnt += batch_size

					z_hat = z + (torch.rand((batch_size, 1)).to(device) * (r - l) + l) * F.normalize(torch.randn(batch_size, nz).to(device), dim=1)
					x_hat = gan2cls(G(z_hat), dataset, input_size, device)
					y_hat = model(x_hat if x_hat.shape[1] == 3 else x_hat.repeat(1, 3, 1, 1)).argmax(dim=1, keepdim=True)

					mask = (y_hat != t).view(-1)
					x_hat = x_hat[mask]
					z_hat = z_hat[mask]

					for i in range(z_hat.size(0)):
						x_ast.append(x_hat[i].detach())
						z_ast.append(z_hat[i].detach())

				if len(x_hat) == 0:
					l = (l + r) / 2
				else:
					minval = -1
					minidx = -1

					for i in range(len(z_ast)):
						val = torch.norm(z_ast[i] - z)
						if minidx < 0 or val < minval:
							minidx = i
							minval = val
					
					x_ast_opt = x_ast[minidx]
					z_ast_opt = z_ast[minidx]
					l = 0
					r = minval
				
			itr = 0
			while itr < max_itr and r > 0:
				x_ast = []
				z_ast = []

				l = max(0, r - dr)

				cnt = 0
				while cnt < N:
					cnt += batch_size

					z_hat = z + (torch.rand((batch_size, 1)).to(device) * (r - l) + l) * F.normalize(torch.randn(batch_size, nz).to(device), dim=1)
					x_hat = gan2cls(G(z_hat), dataset, input_size, device)
					y_hat = model(x_hat if x_hat.shape[1] == 3 else x_hat.repeat(1, 3, 1, 1)).argmax(dim=1, keepdim=True)

					mask = (y_hat != t).view(-1)
					x_hat = x_hat[mask]
					z_hat = z_hat[mask]

					for i in range(z_hat.size(0)):
						x_ast.append(x_hat[i].detach())
						z_ast.append(z_hat[i].detach())

				if len(z_ast) == 0:
					itr += 1
					r = r - dr
				else:
					minval = -1
					minidx = -1

					for i in range(len(z_ast)):
						val = torch.norm(z_ast[i] - z)
						if minidx < 0 or val < minval:
							minidx = i
							minval = val
					
					x_ast_opt = x_ast[minidx]
					z_ast_opt = z_ast[minidx]
					l = 0
					r = minval

			if x_ast_opt is None:
				return None
			else:
				return x_ast_opt.unsqueeze(0)

		else:
			raise NotImplementedError


def main():
	opt = UnconditionalLSAOptions().parse()

	# dataset
	dataset = get_dataset(opt.dataset, train=opt.use_train, input_size=opt.input_size, normalize=True, augment=False)
	labels = get_labels(opt.dataset)
	opt.num_classes = len(labels)
	loader = DataLoader(dataset, batch_size=1, shuffle=True)

	# model
	model = get_classifier(opt.arch, opt.num_classes).to(opt.device)
	model.load_state_dict(torch.load(opt.weight))
	model.eval()

	# wgan-gp with inverter
	G, C, I = get_wgan_gp_inverter(opt.nz, opt.nc, opt.nf, opt.gan_size)
	load_wgan_gp_inverter(G, C, I, opt.gan_dir)
	G, C, I = G.to(opt.device), C.to(opt.device), I.to(opt.device)
	G.eval()
	C.eval()
	I.eval()

	# make directories
	for i in range(opt.num_classes):
		for j in range(opt.num_classes):
			os.makedirs(os.path.join(opt.log_dir, '{:03d}/{:03d}'.format(i, j)), exist_ok=True)

	cnt = 0
	total = 0

	for itr, (x, t) in enumerate(loader):
		x, t = x.to(opt.device), t.to(opt.device)

		init_pred = model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)).argmax()

		if init_pred.item() != t.item():
			continue

		perturbed_x = unconditional_lsa_attack(
			x, t, model, G, C, I, opt.dataset, 
			opt.method, opt.max_r, opt.dr, opt.batch_size, 
			opt.max_itr, opt.N, opt.input_size, opt.gan_size, opt.device
		)
		
		if perturbed_x is None:
			continue

		final_pred = model(perturbed_x if perturbed_x.shape[1] == 3 else perturbed_x.repeat(1, 3, 1, 1)).argmax()

		total += 1
		if final_pred.item() != t.item():
			cnt += 1

			save_image(
				torch.cat((unnormalize(x.cpu(), opt.dataset), unnormalize(perturbed_x.cpu(), opt.dataset)), dim=0),
				os.path.join(opt.log_dir, '{:03d}/{:03d}/{:05d}.png'.format(init_pred.item(), final_pred.item(), cnt)),
				padding=0
			)

		if cnt % 10 == 0:
			sys.stdout.write('\r\033[K{:d} adversarial examples are found from {:d} samples.'.format(cnt, total))
			sys.stdout.flush()

	sys.stdout.write('\r\033[K{:d} adversarial examples are found from {:d} samples.\n'.format(cnt, total))
	sys.stdout.write('success rate {:.2f}\n'.format(float(cnt)/float(total)))
	sys.stdout.flush()


if __name__ == '__main__':
	main()
