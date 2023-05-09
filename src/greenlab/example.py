import torch


def add_one(number):
    return number + 1


def add_one_tensor(tensor):
    return tensor + torch.ones_like(tensor)
