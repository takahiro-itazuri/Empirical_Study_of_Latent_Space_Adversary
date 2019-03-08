import os
import sys
import random

import torch
import torchvision
from torch.nn import functional as F
from torchvision.utils import save_image

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from utils import *
from models import *
from wgan_gp.options import TestOptions


def main():
	opt = TestOptions().parse()

	labels = get_labels(opt.dataset)
	opt.num_classes = len(labels)

	# model
	G, _ = get_wgan_gp(opt.nz, opt.nc, opt.nf, opt.image_size, opt.condition, opt.num_classes)
	G = G.to(opt.device)
	G.load_state_dict(torch.load(opt.weight))
	if torch.cuda.device_count() > 1:
		G = torch.nn.DataParallel(G)

	# make directories
    if opt.condition:
        for i in range(opt.num_classes):
            os.makedirs(os.path.join(opt.log_dir, '{:03d}'.format(i)), exist_ok=True)
    else:
        os.makedirs(opt.log_dir)

	cnt = 0
	while True:
		if cnt >= opt.num_samples:
			break
		elif cnt + opt.batch_size >= opt.num_samples:
			batch_size = opt.num_samples - cnt
		else:
			batch_size = opt.batch_size

		z = torch.randn((batch_size, opt.nz)).to(opt.device)
        if opt.condition:
            t = torch.tensor([random.randrange(0, opt.num_classes) for i in range(batch_size)]).to(opt.device)
        else:
            t = None

		samples = G(z, t).detach()
		for i in range(samples.shape[0]):
			sample = F.interpolate(samples[i].unsqueeze(0), size=opt.output_size, mode='bilinear')

            if t != None:
                output_path = os.path.join(opt.log_dir, '{:03d}/{:05d}.png'.format(t[i].item(), cnt))
            else:
                output_path = os.path.join(opt.log_dir, 'imgs/{:05d}.png'.format(cnt))

            save_image(
                unnormalize(sample.cpu(), opt.dataset)
                output_path, padding=0
            )

			cnt += 1
			if cnt % 100 == 0:
				sys.stdout.write('\r\033[Kgenerating... [{:d}/{:d}]'.format(cnt, opt.num_samples))
				sys.stdout.flush()

	sys.stdout.write('\r\033[Kdone!\n'.format(cnt, opt.num_samples))
	sys.stdout.flush()


if __name__ == '__main__':
	main()
