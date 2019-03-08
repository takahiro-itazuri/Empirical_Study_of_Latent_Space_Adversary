"""
This code is the implementation of "Generating Natural Adversarial Examples".
(https://openreview.net/pdf?id=H1BLjgZCb)

The original implemention by TensorFlow is available.
(https://github.com/zhengliz/natural-adversary)

This is implemented with reference to following links.
- https://github.com/zhengliz/natural-adversary
- https://github.com/hjbahng/natural-adversary-pytorch
- https://github.com/igul222/improved_wgan_training
"""

import os
import sys
import math

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import grad
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter


base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *
from models import *
from wgan_gp_inverter.options import TrainOptions
from wgan_gp_inverter.utils import *


def main():
	opt = TrainOptions().parse()

	writer = SummaryWriter(os.path.join(opt.log_dir, 'runs'))

	# dataset
	dataset = get_dataset(opt.dataset, train=True, input_size=opt.image_size, normalize=False, augment=False)
	loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

	# model
	G, C, I = get_wgan_gp_inverter(opt.nz, opt.nc, opt.nf, opt.image_size)

	G = G.to(opt.device)
	C = C.to(opt.device)
	I = I.to(opt.device)

	if torch.cuda.device_count() > 1:
		G = torch.nn.DataParallel(G)
		C = torch.nn.DataParallel(C)
		I = torch.nn.DataParallel(I)

	# optimizer
	G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
	C_optimizer = optim.Adam(C.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
	I_optimizer = optim.Adam(I.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

	if opt.resume:
		opt.last_epoch = load_checkpoint(G, C, I, G_optimizer, C_optimizer, I_optimizer, opt.resume, os.path.join(opt.log_dir, 'checkpoint.pth.tar'))
	else:
		opt.last_epoch = 0

	# fixed z & image
	fix_z = torch.randn((36, opt.nz)).to(opt.device)
	fix_image = next(iter(loader))[0][:18].to(opt.device)

	# train
	itr = opt.last_epoch
	running_EMD = 0.0
	running_I_loss = 0.0
	timer = Timer(opt.num_itrs, opt.last_epoch)

	while True:
		for i, (x_real, t) in enumerate(loader):
			if x_real.size(0) != opt.batch_size:
				break

			x_real = normalize(x_real.to(opt.device), None, opt.device)
			z = torch.randn((opt.batch_size, opt.nz)).to(opt.device)


			# === update C network === #
			G.eval()
			C.train()
			I.eval()

			# EMD
			x_fake = G(z).detach()
			EMD = C(x_real).mean() - C(x_fake).mean()

			# gradient penalty
			alpha = torch.rand((opt.batch_size, 1, 1, 1)).to(opt.device)
			x_hat = alpha * x_real.data + (1 - alpha) * x_fake.data
			x_hat.requires_grad_()
			C_hat = C(x_hat)
				
			gradients = grad(outputs=C_hat, inputs=x_hat, grad_outputs=torch.ones(C_hat.size()).to(opt.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
			GP = opt.lambda_gp * ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

			C_loss = -EMD + GP

			C_optimizer.zero_grad()
			C_loss.backward()
			C_optimizer.step()
				
			# === update I network === #
			G.eval()
			C.eval()
			I.train()

			z_recon_loss = torch.mean((I(G(z).detach()) - z)**2)
			x_recon_loss = torch.mean((G(I(x_real)) - x_real)**2)

			I_loss = x_recon_loss + opt.lambda_z * z_recon_loss

			I_optimizer.zero_grad()
			I_loss.backward()
			I_optimizer.step()

			if ((i + 1) % opt.num_critic) == 0:
				itr += 1

				# === update G network === #
				G.train()
				C.eval()
				I.eval()

				G_loss = -C(G(z)).mean()

				G_optimizer.zero_grad()
				G_loss.backward()
				G_optimizer.step()

				# standard output
				timer.step()
				elapsed_time = timer.get_elapsed_time()
				estimated_time = timer.get_estimated_time()

				G.eval()
				running_EMD += EMD.item()
				running_I_loss += I_loss.item()
				commandline_output = ''

				# save model
				if itr % 10 == 0:
					samples = G(fix_z)
					recon_image = G(I(fix_image))
					save_image(unnormalize(samples, None, opt.device), os.path.join(opt.log_dir, 'latest.png'), nrow=6)
					commandline_output = '\r\033[K[itr {:d}] EMD: {:.4f}, I_loss: {:.4f}, elapsed time: {}, estimated time: {}'.format(int(itr), EMD.item(), I_loss.item(), elapsed_time, estimated_time)

					if itr % 1000 == 0:
						save_image(unnormalize(samples, None, opt.device), os.path.join(opt.log_dir, '{:06d}itr.png'.format(int(itr))), nrow=6)
						if writer is not None:
							writer.add_image('generated samples', make_grid(unnormalize(samples, None, opt.device), nrow=6), global_step=itr)
							writer.add_image('reconstructed samples', make_grid(torch.cat([unnormalize(fix_image, None, opt.device), unnormalize(recon_image, None, opt.device)]), nrow=6), global_step=itr)
							writer.add_scalars('Loss', {'EMD': running_EMD / 1000, 'I_loss': running_I_loss / 1000}, global_step=itr)
						commandline_output = '\r\033[K[itr {:d}] EMD: {:.4f}, I_loss: {:.4f}, elapsed time: {}, estimated time: {}\n'.format(int(itr), running_EMD / 1000, running_I_loss / 1000, elapsed_time, estimated_time)
						running_EMD = 0.0
						running_I_loss = 0.0

					del samples
					del recon_image

				save_checkpoint(G, C, I, G_optimizer, C_optimizer, I_optimizer, itr, os.path.join(opt.log_dir, 'checkpoint.pth.tar'))

					
				sys.stdout.write(commandline_output)
				sys.stdout.flush()

			if itr == opt.num_itrs:
				break
		
		if itr == opt.num_itrs:
			break

	save_model(G, os.path.join(opt.log_dir, 'G_weight_final.pth'))
	save_model(C, os.path.join(opt.log_dir, 'C_weight_final.pth'))
	save_model(I, os.path.join(opt.log_dir, 'I_weight_final.pth'))


if __name__ == '__main__':
	main()
