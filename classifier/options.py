import os
import sys
import random
import argparse

import torch
from torchvision import models

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from adversarial_examples.utils import get_eps, get_alpha
from classifier.utils import get_lr


model_names = ['alexnet', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
dataset_names = ['mnist', 'svhn', 'cifar10', 'cifar100', 'stl10', 'imagenet']
mode_names = ['normal', 'half', 'adversarial']
attack_names = ['fgsm', 'pgd']


class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		# model
		parser.add_argument('-a', '--arch', type=str, required=True, choices=model_names, help='model architecture: ' + ' | ' .join(model_names), metavar='ARCH')
		parser.add_argument('--pretrained', action='store_true', default=False, help='use pre-trained model')
		# dataset
		parser.add_argument('-d', '--dataset', type=str, default='imagenet', choices=dataset_names, help='dataset: ' + ' | '.join(dataset_names), metavar='DATASET')
		parser.add_argument('-j', '--num_workers', type=int, default=4, help='number of workers for data loading')
		parser.add_argument('-N', '--batch_size', type=int, default=256, help='batch size')
		# GPU
		parser.add_argument('--cuda', action='store_true', default=None, help='enable GPU')
		# log
		parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
		parser.add_argument('-l', '--log_dir', type=str, required=True, help='log directory')
		parser.add_argument('-r', '--result', type=str, default='result.json', help='result json path')
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

		# GPU
		if opt.cuda and torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			opt.device = 'cuda'
		else:
			opt.cuda = False
			opt.device = 'cpu'

		self.opt = opt
		return self.opt
			

class TrainOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# mode
		parser.add_argument('-m', '--mode', type=str, required=True, choices=mode_names, help='train mode: ' + ' | '.join(mode_names), metavar='MODE')
		# model
		parser.add_argument('-c', '--checkpoint', type=str, default=None, help='checkpoint path for resume')
		parser.add_argument('-w', '--weight', type=str, default=None, help='model weight path')
		# dataset
		parser.add_argument('--num_samples', type=int, default=-1, help='number of samples (-1 means all)')
		# hyperparameter
		parser.add_argument('--last_epoch', type=int, default=0, help='last epoch')
		parser.add_argument('--num_epochs', type=int, default=90, help='number of epochs')
		parser.add_argument('--lr', type=float, default=None, help='initial learning rate')
		parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
		parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
		# scheduler
		parser.add_argument('--step_size', type=int, default=30, help='step size for scheduler')
		parser.add_argument('--gamma', type=float, default=0.1, help='gamma for scheduler')
		# adversarial examples
		parser.add_argument('--attack', type=str, default=None, choices=attack_names, help='adversarial attack method: ' + ' | '.join(attack_names), metavar='ATTACK')
		parser.add_argument('--p', type=int, default=-1, help='type of norm (-1 means l_infty norm)')
		parser.add_argument('--eps', type=float, default=None, help='perturbation size')
		parser.add_argument('--num_steps', type=int, default=5, help='number of steps for PGD')
		parser.add_argument('--alpha', type=float, default=None, help='perturbation size for single step')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		if opt.mode != 'normal':
			if opt.eps == None:
				opt.eps = get_eps('train', opt.p, opt.dataset)
			if opt.alpha == None:
				opt.alpha = get_alpha(opt.eps, opt.num_steps)

		if opt.lr == None:
			opt.lr = get_lr(opt.arch)

		self.opt = opt
		self.print_options(opt)
		return self.opt
			


class ValidateOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# model
		parser.add_argument('-w', '--weight', type=str, default=None, help='model weight path')
		# dataset
		parser.add_argument('--use_train', action='store_true', default=False, help='use train set')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		self.opt = opt
		self.print_options(opt)
		return self.opt
