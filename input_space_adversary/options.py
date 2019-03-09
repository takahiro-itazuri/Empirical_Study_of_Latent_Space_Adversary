import os
import sys
import argparse

import torch

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from input_space_adversary.utils import get_eps, get_alpha


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
		# dataset
		parser.add_argument('-d', '--dataset', type=str, default='imagenet', choices=dataset_names, help='dataset: ' + ' | '.join(dataset_names), metavar='DATASET')
		parser.add_argument('--use_train', action='store_true', default=False, help='use train dataset')
		# output
		parser.add_argument('-l', '--log_dir', type=str, required=True, help='log directory')
		parser.add_argument('--save_image', action='store_true', default=False, help='save image or not')
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

		# input image size
		if opt.arch == 'lenet':
			opt.input_size = 32
		else:
			opt.input_size = 224

		# GPU
		if opt.cuda and torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			opt.device = 'cuda'
		else:
			opt.device = 'cpu'

		self.opt = opt
		return self.opt


class FGSMOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		parser.add_argument('--p', type=int, default=-1, help='type of norm (-1 means l_infty norm)')
		parser.add_argument('--eps', type=float, default=None, help='perturbation size')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		if opt.eps == None:
			opt.eps = get_eps('train', opt.p, opt.dataset)

		self.opt = opt
		self.print_options(opt)
		return self.opt
	

class PGDOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		parser.add_argument('--p', type=int, default=-1, help='type of norm (-1 means l_infty norm)')
		parser.add_argument('--eps', type=float, default=None, help='perturbation size')
		parser.add_argument('--alpha', type=float, default=None, help='perturbation size of single step')
		parser.add_argument('--num_steps', type=int, default=10, help='number of steps')
		parser.add_argument('--restarts', type=int, default=10, help='times of restarting')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		if opt.eps == None:
			opt.eps = get_eps('train', opt.p, opt.dataset)

		if opt.alpha == None:
			opt.alpha = get_alpha(opt.eps, opt.num_steps)

		self.opt = opt
		self.print_options(opt)
		return self.opt


class DeepFoolOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		parser.add_argument('--overshoot', type=float, default=0.02, help='overshoot value')
		parser.add_argument('--num_candidates', type=int, default=-1, help='number of candidate classes (calculated from the class with higher likelihood)')
		parser.add_argument('--max_itr', type=int, default=10, help='number of maximum iteration')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		self.opt = opt
		self.print_options(opt)
		return self.opt