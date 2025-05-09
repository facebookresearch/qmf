# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import qmf.dataStructures as ds
import torch
from qmf.numpyTorch import getDevice


def initRes(p, numBS, numVertices):
    torch.manual_seed(p.seed)
    rndC = torch.randn(p.numColumnB, numVertices)

    return ds.Res(
        B=(p.initB * torch.randn(numBS * 3, p.numColumnB))
        .clone()
        .float()
        .to(getDevice())
        .requires_grad_(),
        C=(p.initC * rndC).clone().float().to(getDevice()).requires_grad_(),
    )
