import os
import argparse

import torch

model_names = ['lenet', 'alexnet', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
dataset_names = ['mnist', 'svhn', 'cifar10', 'cifar100', 'lsun', 'imagenet']


class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		# model
		parser.add_argument('-a', '--arch', type=str, required=True, choices=model_names, help='model architecture: ' + ' | ' .join(model_names), metavar='ARCH')
		parser.add_argument('-w', '--weight', type=str, default=None, help='model weight path')
		parser.add_argument('--pretrained', action='store_true', default=False, help='use pre-trained model')
		parser.add_argument('-g', '--gan_dir', type=str, required=True, help='directory to GAN weight')
		# dataset
		parser.add_argument('-d', '--dataset', type=str, default='imagenet', choices=dataset_names, help='dataset: ' + ' | '.join(dataset_names), metavar='DATASET')
		parser.add_argument('--use_train', action='store_true', default=False, help='use train dataset')
		# output
		parser.add_argument('-l', '--log_dir', type=str, required=True, help='log directory')
		parser.add_argument('-N', '--num_samples', type=int, default=-1, help='number of samples (-1 means all samples)')
		# GPU
		parser.add_argument('--cuda', action='store_true', default=False, help='enable GPU')
		self.initialize = True
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

		# input image size of classifier
		if opt.arch == 'lenet':
			opt.input_size = 32
		else:
			opt.input_size = 224

		# output image size of gan
		if opt.dataset == 'mnist':
			opt.gan_size = 28
		elif opt.dataset in ['svhn', 'cifar10']:
			opt.gan_size = 32
		elif opt.dataset == 'lsun':
			opt.gan_size = 64

		# GPU
		if opt.cuda and torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			opt.device = 'cuda'
		else:
			opt.device = 'cpu'

		self.opt = opt
		return self.opt


class ConditionalLSAOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# hyperparameter
		parser.add_argument('--alpha', type=float, default=0.01, help='learning rate')
		parser.add_argument('--eps_z', type=float, default=0.001, help='epsilon for latent perturbation (soft constraint)')
		parser.add_argument('--lambda1', type=float, default=10.0, help='coefficient of latent perturbation loss')
		parser.add_argument('--lambda2', type=float, default=10.0, help='coefficient of image perturbation loss')
		parser.add_argument('--max_itr', type=int, default=250, help='max iteration')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		if opt.dataset == 'mnist':
			opt.nz = 64
			opt.nc = 1
			opt.nf = 32
		else:
			opt.nz = 128
			opt.nc = 3
			opt.nf = 128

		if opt.num_samples <= 0:
			raise ValueError('num_samlpes should be more than 0.')

		self.opt = opt
		self.print_options(opt)
		return self.opt
			

class UnconditionalLSAOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		parser.add_argument('--method', type=str, default='hybrid_shrinking', help='search method')
		parser.add_argument('--max_itr', type=int, default=10, help='max iteration')
		parser.add_argument('--max_r', type=float, default=3.0, help='max perturbation size')
		parser.add_argument('--dr', type=float, default=0.3, help='perturbaion size for single step')
		parser.add_argument('--N', type=int, default=1000, help='number of samples for each iteration')
		parser.add_argument('--batch_size', type=int, default=500, help='batch size')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)
		
		if opt.dataset == 'mnist':
			opt.nz = 64
			opt.nc = 1
			opt.nf = 64
		elif opt.dataset == 'svhn':
			opt.nz = 64
			opt.nc = 3
			opt.nf = 64
		else:
			opt.nz = 128
			opt.nc = 3
			opt.nf = 128

		self.opt = opt
		self.print_options(opt)
		return self.opt
			
		
