# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import time

import qmf.dataStructures as ds
import termcolor as fn
import torch


def printIntermediateResult(i, B, C, D, X, geo, loss, st):
    trunc_err = D.abs().max().item()

    torch.cuda.synchronize()
    numNz = (C.abs() > 1e-4).count_nonzero(dim=0)
    CnzMax = numNz.max()
    CnzMin = numNz.min()

    print(
        fn.clearLine
        + f"{i:05d}({time.time() - st:.3f}) "
        + f"loss:{loss.item():.5e} "
        + f"maxD:{trunc_err:.5e} "
        + f"Nnz:{(B.abs() > 1e-4).count_nonzero().item():_}|{(C.abs() > 1e-4).count_nonzero().item():_} "
        + f"Cnz:{CnzMin}-{CnzMax}",
        end="\r",
    )


def cullMatricesBC(p, pre, B, C):
    numNz = p.numNz

    # here we keep maximum numNz.maxNumPerCCol non-zero values per C matrix column
    cColumnCutOff = torch.topk(C.abs(), numNz.maxNumPerCCol + 1, dim=0).values[-1, :]
    cColPruned = (C.abs() > cColumnCutOff) * C
    # we keep numNz.all number of non-zero values for both matrices B and C
    cAmp = cColPruned * pre.cCutMultiplyer
    cAndB = torch.cat((B.flatten(), cAmp.flatten()))
    cCutoff = torch.topk(cAndB.abs(), numNz.all).values[-1]
    bPruned = (B.abs() >= cCutoff) * B
    B.copy_(bPruned)
    cPruned = (cAmp.abs() >= cCutoff) * C
    C.copy_(cPruned)


def train(optimizer, p, geo, pre, res):
    B, C = res
    A = geo.A
    laplacian = geo.laplacian

    lossHistory = []

    st = time.time()

    for i in range(p.numIterations):
        X = B @ C
        D = A - X

        laplacianD = laplacian @ D.t()

        loss0 = D.pow(p.power).mean()
        loss2 = laplacianD.square().mean()
        loss = loss0 + p.alpha * loss2

        lossHistory.append((float(loss), D.abs().max().item()))
        if i % 200 == 0:
            printIntermediateResult(i, B, C, D, X, geo, loss, st)
            st = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cullMatricesBC(p, pre, B, C)

    print()
    return ds.Res(B, C), lossHistory
