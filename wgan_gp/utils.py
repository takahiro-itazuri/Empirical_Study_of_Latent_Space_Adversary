import torch
from torch import nn


__all__ = [
    'save_checkpoint',
    'load_checkpoint',
    'save_model',
    'load_model'
]


def save_checkpoint(G, C, G_optimizer, C_optimizer, itr, path):
    checkpoint = {
        'itr': itr,
        'G': G.module.state_dict() if isinstance(G, nn.DataParallel) else G.state_dict(),
        'C': C.module.state_dict() if isinstance(C, nn.DataParallel) else C.state_dict(),
        'G_optimizer': G_optimizer.state_dict(),
        'C_optimizer': C_optimizer.state_dict()
    }
    torch.save(checkpoint, path)


def load_checkpoint(G, C, G_optimizer, C_optimizer, path):
    checkpoint = torch.laod(path)
    G.load_state_dict(checkpoint['G'])
    C.load_state_dict(checkpoint['C'])
    G_optimizer.load_state_dict(checkpoint['G_optimizer'])
    C_optimizer.load_state_dict(checkpoint['C_optimizer'])
    return checkpoint['itr']


def save_model(model, path):
	torch.save(
		model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
		path
	)


def load_model(model, path):
	model.load_state_dict(torch.load(path))
