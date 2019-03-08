import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

# ignore exif warning
import warnings
warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *
from classifier.options import ValidateOptions
from classifier.utils import *

def main():
	opt = ValidateOptions().parse()

	# dataset
	dataset = get_dataset(opt.dataset, train=opt.use_train, input_size=opt.input_size)	
	loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	labels = get_labels(opt.dataset)
	opt.num_classes = len(labels)

	# model
	model = get_classifier(opt.arch, opt.num_classes, opt.pretrained).to(opt.device)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	if opt.weight != None:
		model.load_state_dict(torch.load(opt.weight))

	# criterion
	criterion = nn.CrossEntropyLoss()

	# validate
	loss, acc1, acc5 = validate(model, loader, criterion, opt)

	sys.stdout.write(
		'loss {:.4f}, '
		'acc1 {:.2f}%, '
		'acc5 {:.2f}%\n'.format(
			loss, acc1, acc5,
		)
	)
	sys.stdout.flush()


if __name__ == '__main__':
	main()