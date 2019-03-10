import os
import sys
import json

import torch
from torch import nn, optim

torch.backends.cudnn.benchmark=True

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *
from adversarial_examples import *

__all__ = [
	'train', 
	'validate', 
	'load_checkpoint', 
	'save_checkpoint', 
	'load_model',
	'save_model',
	'save_result',
	'get_lr'
]


def train(model, loader, criterion, optimizer, scheduler, opt):
	loss_meter = AverageMeter()
	acc1_meter = AverageMeter()
	acc5_meter = AverageMeter()

	model.train()

	for itr, (x, t) in enumerate(loader):
		if opt.cuda:
			x = x.to(opt.device, non_blocking=True)
			t = t.to(opt.device, non_blocking=True)

		if opt.mode != 'normal':
			model.eval()

			if opt.attack == 'fgsm':
				perturbed_x = fgsm_attack(x, t, model, opt.dataset, opt.eps, p=opt.p, device=opt.device)
			elif opt.attack == 'pgd':
				perturbed_x = pgd_attack(x, t, model, opt.dataset, opt.eps, alpha=opt.alpha, num_steps=opt.num_steps, restarts=1, randomize=True, p=opt.p, device=opt.device)
			else:
				raise NotImplementedError

			if opt.mode == 'adversarial':
				x = perturbed_x
			elif opt.mode == 'half':
				x[:int(opt.batch_size/2)] = perturbed_x[:int(opt.batch_size/2)]

			model.train()		

		y = model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1))

		loss = criterion(y, t)
		acc1, acc5 = accuracy(y, t, topk=(1,5))

		loss_meter.update(loss.item(), x.size(0))
		acc1_meter.update(acc1.item(), x.size(0))
		acc5_meter.update(acc5.item(), x.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if itr % opt.print_freq == 0:
			sys.stdout.write(
				'\r\033[K'
				'[train mode] '
				'itr [{:d}/{:d}] '
				'loss {:.4f}, '
				'acc1 {:.2f}%, '
				'acc5 {:.2f}%'.format(
					itr, len(loader), loss.item(), acc1, acc5
				)
			)

	return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


def validate(model, loader, criterion, opt):
	loss_meter = AverageMeter()
	acc1_meter = AverageMeter()
	acc5_meter = AverageMeter()

	model.eval()

	with torch.no_grad():
		for itr, (x, t) in enumerate(loader):
			if opt.cuda:
				x = x.to(opt.device, non_blocking=True)
				t = t.to(opt.device, non_blocking=True)
			
			y = model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1))

			loss = criterion(y, t)
			acc1, acc5 = accuracy(y, t, topk=(1,5))

			loss_meter.update(loss.item(), x.size(0))
			acc1_meter.update(acc1.item(), x.size(0))
			acc5_meter.update(acc5.item(), x.size(0))

			if itr % opt.print_freq == 0:
				sys.stdout.write(
					'\r\033[K'
					'[val mode] '
					'itr [{:d}/{:d}] '
					'loss {:.4f}, '
					'acc1 {:.2f}%, '
					'acc5 {:.2f}%'.format(
						itr, len(loader), loss.item(), acc1, acc5
					)
				)

	return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


def load_model(model, path):
	model.load_state_dict(torch.load(path))


def save_model(model, path):
	torch.save(
		model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
		path
	)


def load_checkpoint(model, optimizer, path):
	checkpoint = torch.load(path)
	model.load_state_dict(checkpoint['model'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	return checkpoint['epoch']


def save_checkpoint(model, optimizer, epoch, path):
	checkpoint = {
		'epoch': epoch,
		'model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
		'optimizer': optimizer.state_dict()
	}
	torch.save(checkpoint, path)


def save_result(result, log_dir, filename):
	path = os.path.join(log_dir, filename)
	dir = os.path.dirname(path)
	os.makedirs(dir, exist_ok=True)

	with open(path, 'w') as f:
		f.write(json.dumps(result, indent=4))


def accuracy(output, target, topk=(1,)):
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, dim=1) # top-k index: size (B, k)
		pred = pred.t() # size (k, B)
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		acc = []
		for k in topk:
			correct_k = correct[:k].float().sum()
			acc.append(correct_k.item() * 100.0 / batch_size)
		return acc


def get_lr(arch):
	"""
	Returns learning rate given model architecture

	Args:
	arch (str): model architecture name
	"""
	if arch in ['lenet', 'alexnet', 'vgg16_bn', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
		return 0.1
	elif arch in ['vgg16', 'vgg19']:
		return 0.01
	else:
		raise NotImplementedError
