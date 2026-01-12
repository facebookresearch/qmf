# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import qmf.init as init
import qmf.train as train
import termcolor as fn
import torch


def chooseBestSeed(p0, geo, pre, numIterations, numSeeds):
    p = p0._replace(numIterations=int(numIterations), numBits=0)

    bestSeed = (0, float("inf"))
    for seed in range(p0.seed, p0.seed + numSeeds):
        p = p._replace(seed=seed)
        res = init.initRes(p, geo.numBS, geo.numVertices)
        optimizer = torch.optim.Adam([res.B, res.C], lr=p0.lr, betas=(0.9, 0.9))
        loss, maxD = train.train(optimizer, p, geo, pre, res)[1][-1]
        if loss < bestSeed[1]:
            print(fn.up + fn.clearToEnd + fn.up)

            print(
                f"seed {seed:5} {fn.bgGreen}{fn.fgBlack} loss: {loss:7.5} maxD: {maxD:7.5}{fn.reset}"
            )
            bestSeed = (seed, loss)
        else:
            print(fn.up + fn.up)
    return bestSeed[0]
