# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import termcolor as fn
import torch


def tensorQuantization(m, numBits):
    if numBits == 0:
        return m, 0.0, 0.0

    maxInt = 2**numBits - 1

    mMin = m.min()
    m = m - mMin
    mRange = m.max()

    qM = torch.round(m / mRange * maxInt).type(torch.int64)
    qM[qM > maxInt] = maxInt  # to avoid integer overflow due to rounding up
    return qM, mMin, mRange


def tensorDeQuantization(im, mMin, mRange, numBits):
    maxInt = 2**numBits - 1
    return mMin + im.type(torch.float32) / maxInt * mRange


def tensorQuantizationAndPack(m, numBits):
    if numBits == 11_11_10:
        if m.dim() > 1:
            print(
                f"{fn.fgRed}tensorQuantizationAndPack doesn't support tensors with dim() > 1{fn.reset}"
            )
        m11, mMin, mRange = tensorQuantization(m, 11)
        m10, mMin, mRange = tensorQuantization(m, 10)

        m11.resize_((m11.shape[0] + 2) // 3 * 3)
        m10.resize_((m10.shape[0] + 2) // 3 * 3)
        packedM = (m10[2::3] << 22) | (m11[1::3] << 11) | (m11[::3])

        return packedM, mMin, mRange

    else:
        return tensorQuantization(m, numBits)


def tensorQuantizationDeQuantization(m, numBits):
    if numBits == 0:
        return m
    elif numBits == 11_11_10:
        print(
            f"{fn.fgRed}tensorQuantizationDeQuantization doesn't support 11_11_10, 11 bits will be used instead{fn.reset}"
        )
        numBits = 11

    qM, mMin, mRange = tensorQuantization(m, numBits)
    restoredM = tensorDeQuantization(qM, mMin, mRange, numBits)
    restoredM[m == 0.0] = 0.0  # we use sparse matrices so zeros stays zeros
    return restoredM
