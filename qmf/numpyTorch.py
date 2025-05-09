# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import torch


def getDevice():
    return "cuda" if torch.cuda.is_available() else "cpu"


def toNumpy(t):
    return t.detach().cpu().numpy()


def toTorch(t):
    return torch.from_numpy(t).to(getDevice())


def toTorchCpu(t):
    return torch.from_numpy(t).to("cpu")
