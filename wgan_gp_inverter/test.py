import os
import sys
import random

import torch
from torch.nn import functional as F

from options import TestOptions

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from utils import *
from models import *


def main():
	opt = TestOptions().parse()

	# model
	G, _ , _= get_wgan_gp_inverter(opt.nz, opt.nc, opt.nf, opt.image_size)
	G = G.to(opt.device)
	G.load_state_dict(torch.load(os.path.join(opt.log_dir, 'G_weight_final.pth')))
	if torch.cuda.device_count() > 1:
		G = torch.nn.DataParallel(G)

	# make directories
	os.makedirs(os.path.join(opt.output_dir, '{:03d}/{:03d}'.format(0, 0)), exist_ok=True)

	cnt = 0
	while True:
		if cnt >= opt.num_samples:
			break
		elif cnt + opt.batch_size >= opt.num_samples:
			batch_size = opt.num_samples - cnt
		else:
			batch_size = opt.batch_size

		z = torch.randn((batch_size, opt.nz)).to(opt.device)

		samples = G(z).detach()
		for i in range(samples.shape[0]):
			if opt.output_size is not None:
				sample = F.interpolate(samples[i].unsqueeze(0), size=opt.output_size, mode='bilinear')

			output_path_without_ext = os.path.join(opt.output_dir, '{:03d}/{:03d}/{:05d}'.format(0, 0, cnt))

			if opt.output_type == 'both':
				save_as_dual_image(sample, sample, output_path_without_ext)
				save_as_dual_numpydata(sample, sample, output_path_without_ext, opt.output_bit)

			elif opt.output_type == 'image':
				save_as_dual_image(sample, sample, output_path_without_ext)

			elif opt.output_type == 'numpy':
				save_as_dual_numpydata(sample, sample, output_path_without_ext, opt.output_bit)
			
			cnt += 1
			if cnt % 100 == 0:
				sys.stdout.write('\r\033[Kgenerating... [{:d}/{:d}]'.format(cnt, opt.num_samples))
				sys.stdout.flush()

	sys.stdout.write('\r\033[Kdone!\n'.format(cnt, opt.num_samples))
	sys.stdout.flush()


if __name__ == '__main__':
	main()