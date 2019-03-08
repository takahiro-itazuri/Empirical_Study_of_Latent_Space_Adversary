import os
import sys
import argparse
import torch


class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		# model
		parser.add_argument('--condition', type=int, default=1, help='0: without condition, 1: with condition')
		# dataset
		parser.add_argument('-d', '--dataset', type=str, required=True, help='mnist | svhn | cifar10 | lsun')
		# log
		parser.add_argument('-l', '--log_dir', type=str, required=True, help='log directory')
		# GPU
		parser.add_argument('--cuda', action='store_true', default=False, help='enable GPU')

		self.initialized = True
		return parser

	def gather_options(self):
		if not self.initialized:
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		message = ''
		message += '---------------------------- Options --------------------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: {}]'.format(str(default))
			message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
		message += '---------------------------- End ------------------------------'
		print(message)

		os.makedirs(opt.log_dir, exist_ok=True)
		with open(os.path.join(opt.log_dir, 'options.txt'), 'wt') as f:
			command = ''
			for k, v in sorted(vars(opt).items()):
				command += '--{} {} '.format(k, str(v))
			command += '\n'
			f.write(command)
			f.write(message)
			f.write('\n')

	def parse(self):
		opt = self.gather_options()

		if opt.dataset == 'mnist':
			opt.image_size = 28
		if opt.dataset == 'svhn' or opt.dataset == 'cifar10':
			opt.image_size = 32
		elif opt.dataset == 'lsun':
			opt.image_size = 64

		if opt.dataset == 'mnist':
			opt.nz = 64
			opt.nc = 1
			opt.nf = 32
		else:
			opt.nz = 128
			opt.nc = 3
			opt.nf = 128

		# GPU
		if opt.cuda and torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			opt.device = 'cuda'
		else:
			opt.device = 'cpu'

		self.opt = opt
		return self.opt


class TrainOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# dataset
		parser.add_argument('-j', '--num_workers', type=int, default=0, help='number of workers')
		# hyperparameter
		parser.add_argument('--batch_size', type=int, default=256, help='batch size')
		parser.add_argument('--num_itrs', type=int, default=50000, help='number of iterations')
		parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for Adam')
		parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
		parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
		parser.add_argument('--num_critic', type=int, default=5, help='number of critic iteration for one generator iteration')
		parser.add_argument('--lambda_gp', type=float, default=10, help='coefficient of gradient penalty')
		# log
		parser.add_argument('--resume', action='store_true', default=False, help='enable resume mode')
		return parser

    def parse(self):
        opt = BaseOptions.parse(self)

        self.opt = opt
        self.print_options(opt)
        return self.opt


class TestOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# model
		parser.add_argument('-w', '--weight', type=str, required=True, help='model weight path')
		# hyperparameter
		parser.add_argument('--batch_size', type=int, default=128, help='batch size')
		# log
		parser.add_argument('--output_size', type=int, default=None, help='output image size')
		parser.add_argument('-N', '--num_samples', type=int, required=True, help='number of samples to generate adversarial examples')
		return parser

    def parse(self):
        opt = BaseOptions.parse(self)

		if opt.output_size == None:
			opt.output_size = opt.image_size

        self.opt = opt
        self.print_options(opt)
        return self.opt
