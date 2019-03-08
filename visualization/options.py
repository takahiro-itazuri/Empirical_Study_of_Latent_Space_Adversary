import os
import argparse

import torch

model_names = ['alexnet', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
dataset_names = ['mnist', 'svhn', 'cifar10', 'cifar100', 'stl10', 'imagenet']
method_names = ['vanilla', 'loss', 'smooth', 'gruided']


class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		# model
		parser.add_argument('-a', '--arch', type=str, required=True, choices=model_names, help='model architecture: ' + ' | ' .join(model_names), metavar='ARCH')
		parser.add_argument('-w', '--weight', type=str, default=None, help='model weight path')
		parser.add_argument('--pretrained', action='store_true', default=False, help='use pre-trained model')
		# dataset
		parser.add_argument('-d', '--dataset', type=str, required=True, choices=dataset_names, help='dataset name: ' + ' | '.join(dataset_names), metavar='DATASET')
		# log
		parser.add_argument('-l', '--log_dir', type=str, required=True, help='log directory')
		# GPU
		parser.add_argument('--cuda', action='store_true', default=False, help='enable GPU')
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

		# GPU
		if opt.cuda and torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			opt.device = 'cuda'
		else:
			opt.device = 'cpu'

		self.opt = opt
		return self.opt


class GradientsOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# method
		parser.add_argument('-m', '--method', type=str, required=True, choices=method_names, help='visualization method: ' + ' | '.join(method_names), metavar='METHOD')
		# dataset
		parser.add_argument('--use_train', action='store_true', default=False, help='use train dataset')
		parser.add_argument('-N', '--num_samples', type=int, default=-1, help='number of samples (-1 means all)')
		# others
		parser.add_argument('-r', '--ratio', type=float, default=3, help='cut off ratio of extreme parts')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		self.opt = opt
		self.print_options(opt)
		return self.opt


class ActivationMaximizationOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# hyperparameter
		parser.add_argument('-N', '--num_filters', type=int, default=5, help='number of filters for each layer')
		parser.add_argument('--num_itrs', type=int, default=30, help='number of iterations')
		parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
		parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
		parser.add_argument('--eps', type=float, default=0.01, help='initial noise size')
		parser.add_argument('-r', '--ratio', type=float, default=3.0, help='cut off ratio of extreme parts')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		self.opt = opt
		self.print_options(opt)
		return self.opt