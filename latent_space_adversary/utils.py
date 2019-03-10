import os
import sys

import torch
from torch.nn import functional as F

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)
from misc import *


__all__ = [
	'load_wgan_gp',
	'load_wgan_gp_inverter',
	'gan2cls',
	'cls2gan'
]


def load_wgan_gp(G, C, gan_dir):
	G.load_state_dict(torch.load(os.path.join(gan_dir, 'G_weight_final.pth')))
	C.load_state_dict(torch.load(os.path.join(gan_dir, 'C_weight_final.pth')))


def load_wgan_gp_inverter(G, C, I, gan_dir):
	G.load_state_dict(torch.load(os.path.join(gan_dir, 'G_weight_final.pth')))
	C.load_state_dict(torch.load(os.path.join(gan_dir, 'C_weight_final.pth')))
	I.load_state_dict(torch.load(os.path.join(gan_dir, 'I_weight_final.pth')))


def gan2cls(x, dataset, size, device='cpu'):
	x = normalize(unnormalize(x, None, device), dataset, device)
	x = F.interpolate(x, size=size, mode='bilinear')
	return x


def cls2gan(x, dataset, size, device='cpu'):
	x = F.interpolate(x, size=size, mode='bilinear')
	x = normalize(unnormalize(x, dataset, device), None, device)
	return x