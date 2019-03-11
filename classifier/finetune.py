import os
import sys
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# ignore exif warning
import warnings
warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *
from classifier.options import FinetuneOptions
from classifier.utils import *


def main():
	opt = FinetuneOptions().parse()

	# dataset
	train_dataset = get_dataset(opt.dataset, train=True, input_size=opt.input_size, num_samples=opt.num_samples)
	val_dataset = get_dataset(opt.dataset, train=False, input_size=opt.input_size)
	train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
	labels = get_labels(opt.dataset)
	opt.num_classes = len(labels)

	# model
	pretrained_num_classes = len(get_labels(opt.pretrained_dataset))
	model = get_classifier(opt.arch, pretrained_num_classes, opt.pretrained)

	if opt.weight is not None:
		load_model(model, opt.weight)
	
	replace_final_fc(opt.arch, model, opt.num_classes)
	model = model.to(opt.device)

	# optimizer
	optimizer = optim.SGD(model.parameters(), opt.lr, momentum=opt.momentum, weight_decay=opt.wd)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

	# load checkpoint for resume
	if opt.checkpoint is not None:
		opt.last_epoch = load_checkpoint(model, optimizer, opt.checkpoint)

	# data parallel
	if opt.cuda and torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	# criterion
	criterion = nn.CrossEntropyLoss()

	# timer
	timer = Timer(opt.num_epochs, opt.last_epoch)

	# train
	writer = SummaryWriter(os.path.join(opt.log_dir, 'runs'))
	for epoch in range(opt.last_epoch+1, opt.num_epochs+1):
		scheduler.step(epoch - 1) # scheduler's epoch is 0-indexed.
		train_loss, train_acc1, train_acc5 = train(model, train_loader, criterion, optimizer, scheduler, opt)
		val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, opt)

		writer.add_scalars('loss', {'train': train_loss, 'val{}': val_loss}, global_step=epoch)
		writer.add_scalars('acc1', {'train': train_acc1, 'val{}': val_acc1}, global_step=epoch)
		writer.add_scalars('acc5', {'train': train_acc5, 'val{}': val_acc5}, global_step=epoch)

		timer.step()

		sys.stdout.write(
			'\r\033[K'
			'epoch [{:d}/{:d}] '
			'train loss {:.4f}, '
			'train acc1 {:.2f}%, '
			'train acc5 {:.2f}%, '
			'val loss {:.4f}, '
			'val acc1 {:.2f}%, '
			'val acc5 {:.2f}%, '
			'elapsed time: {}, '
			'estimated time: {}\n'.format(
				epoch, opt.num_epochs, 
				train_loss, train_acc1, train_acc5,
				val_loss, val_acc1, val_acc5,
				timer.get_elapsed_time(), timer.get_estimated_time()
			)
		)
		sys.stdout.flush()

		save_checkpoint(model, optimizer, epoch, os.path.join(opt.log_dir, 'checkpoint.pth.tar'))
	save_checkpoint(model, optimizer, epoch, os.path.join(opt.log_dir, 'checkpoint.pth.tar'))
	save_model(model, os.path.join(opt.log_dir, 'weight_final.pth'))

	save_result({
		'val': {
			'loss': val_loss,
			'acc1': val_acc1,
			'acc5': val_acc5
		},
		'train': {
			'loss': train_loss,
			'acc1': train_acc1,
			'acc5': train_acc5
		}
	}, opt.log_dir, opt.result)


if __name__ == '__main__':
	main()