"""
This code is the implementation of training WGAN-GP.

This is implemeted with reference to following links.
- https://github.com/znxlwm/pytorch-generative-model-collections
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
from wgan_gp.options import TrainOptions
from wgan_gp.utils import *


def main():
	opt = TrainOptions().parse()

	writer = SummaryWriter(os.path.join(opt.log_dir, 'runs'))

	# dataset
	dataset = get_dataset(opt.dataset, train=True, input_size=opt.image_size)
    labels = get_labels(opt.dataset)
	opt.num_classes = len(labels)
	loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

	# model
	G, C = get_wgan_gp(opt.nz, opt.nc, opt.nf, opt.image_size, opt.condition, opt.num_classes)
	G = G.to(opt.device)
	C = C.to(opt.device)
	if opt.cuda and torch.cuda.device_count() > 1:
		G = torch.nn.DataParallel(G)
		C = torch.nn.DataParallel(C)

	# optimizer
	G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
	C_optimizer = optim.Adam(C.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

	if opt.resume:
		opt.last_epoch = load_checkpoint(G, C, G_optimizer, C_optimizer, os.path.join(opt.log_dir, 'checkpoint.pth.tar'))
    else:
        opt.last_epoch = 0

	# fixed z & t
	num_classes_for_log = min(opt.num_classes, 10)
	num_samples = num_classes_for_log * 10
	if opt.condition:
		fix_z = torch.zeros((num_samples, opt.nz)).to(opt.device)
		for i in range(10):
			fix_z[i * num_classes_for_log] = torch.randn((1, opt.nz))
			for j in range(1, num_classes_for_log):
				fix_z[i * num_classes_for_log + j] = fix_z[i * num_classes_for_log]
		fix_z = fix_z.to(opt.device)

		temp_t = torch.tensor(range(num_classes_for_log))
		fix_t = temp_t.repeat(10)
		fix_t = fix_t.to(opt.device)
	else:
		fix_z = torch.randn((100, opt.nz)).to(opt.device)
		fix_t = None

	# train
	itr = opt.last_epoch
	running_EMD = 0.0
	if opt.condition:
		running_AC_loss = 0.0
		ac_loss = nn.CrossEntropyLoss()
	timer = Timer(opt.num_itrs, opt.last_epoch)

	while True:
		for i, (x_real, t) in enumerate(loader):
			if x_real.size(0) != opt.batch_size:
				break

			x_real = x_real.to(opt.device)
			z = torch.randn((opt.batch_size, opt.nz)).to(opt.device)
			if opt.condition:
				t = t.to(opt.device)

			G.train()
			C.train()

			# === update C network === #
			C_optimizer.zero_grad()

			if opt.condition:
				C_real, y_real = C(x_real)
				C_real = torch.mean(C_real)
				AC_real_loss = ac_loss(y_real, t)

				x_fake = G(z, t).detach()
				C_fake, y_fake = C(x_fake)
				C_fake = torch.mean(C_fake)
				AC_fake_loss = ac_loss(y_fake, t)

				AC_loss = AC_real_loss + AC_fake_loss
			else:
				C_real = C(x_real)
				C_real = torch.mean(C_real)

				x_fake = G(z).detach()
				C_fake = C(x_fake)
				C_fake = torch.mean(C_fake)

			# EMD
			EMD = C_real - C_fake

			# gradient penalty
			alpha = torch.rand((opt.batch_size, 1, 1, 1)).to(opt.device)
			x_hat = alpha * x_real.data + (1 - alpha) * x_fake.data
			x_hat.requires_grad = True

			if opt.condition:
				C_hat, _ = C(x_hat)
			else:
				C_hat = C(x_hat)

			gradients = grad(outputs=C_hat, inputs=x_hat, grad_outputs=torch.ones(C_hat.size()).to(opt.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
			GP = opt.lambda_gp * ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

			if opt.condition:
				C_loss = -EMD + GP + AC_loss
			else:
				C_loss = -EMD + GP
			C_loss.backward()
			C_optimizer.step()


			if ((i + 1) % opt.num_critic) == 0:
				itr += 1

				# === update G === #
				G.train()
				G_optimizer.zero_grad()

				if opt.condition:
					x_fake = G(z, t)
					C_fake, y_fake = C(x_fake)
					C_fake = torch.mean(C_fake)
					AC_fake_loss = ac_loss(y_fake, t)
					G_loss = -C_fake + AC_fake_loss
				else:
					x_fake = G(z)
					C_fake = C(x_fake)
					C_fake = torch.mean(C_fake)
					G_loss = -C_fake

				G_loss.backward()
				G_optimizer.step()

				# log
				G.eval()
				running_EMD += EMD.item()
				if opt.condition:
					running_AC_loss += AC_loss.item()
				commandline_output = ''

				timer.set_step(itr)
				elapsed_time = timer.get_elapsed_time()
				estimated_time = timer.get_estimated_time()

				if itr % 10 == 0:
					samples = G(fix_z, fix_t).detach()
					save_image(unnormalize(samples, opt.dataset, opt.device), os.path.join(opt.log_dir, 'latest.png'), nrow=num_classes_for_log)
					commandline_output = '\r\033[K[itr {:d}] EMD: {:.4f}'.format(int(itr), EMD.item())
					if opt.condition:
						commandline_output += ', AC_loss: {:.4f}'.format(AC_loss.item())
					commandline_output += ', elapsed time: {}, estimated time: {}'.format(elapsed_time, estimated_time)

				if itr % 1000 == 0:
					commandline_output = '\r\033[K[itr {:d}] EMD: {:.4f}'.format(int(itr), running_EMD / 1000)
					if writer is not None:
						writer.add_scalars('Loss', {'EMD': running_EMD / 1000}, global_step=itr)
						writer.add_image('generated samples', make_grid(unnormalize(samples, opt.dataset, opt.device), nrow=num_classes_for_log), global_step=itr)
					running_EMD = 0.0
					if opt.condition:
						commandline_output += ', AC_loss: {:.4f}'.format(running_AC_loss / 1000)
						if writer is not None:
							writer.add_scalars('Loss', {'AC_loss': running_AC_loss / 1000}, global_step=itr)
						running_AC_loss = 0.0
					commandline_output += ', elapsed time: {}, estimated time: {}\n'.format(elapsed_time, estimated_time)

				sys.stdout.write(commandline_output)
				sys.stdout.flush()

				# save model
				if itr % 1000 == 0:
					save_image(unnormalize(samples, opt.dataset, opt.device), os.path.join(opt.log_dir, '{:06d}itr.png'.format(int(itr))), nrow=num_classes_for_log)
				save_checkpoint(G, C, G_optimizer, C_optimizer, itr, os.path.join(opt.log_dir, 'checkpoint.pth.tar'))

			if itr == opt.num_itrs:
				break

		if itr == opt.num_itrs:
			break


    save_model(G, os.path.join(opt.log_dir, 'G_weight_final.pth'))
    save_model(C, os.path.join(opt.log_dir, 'C_weight_final.pth'))


if __name__ == '__main__':
	main()
