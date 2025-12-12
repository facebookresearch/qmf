# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import collections as co

TotalNnzMaxPerCCol = co.namedtuple(
    "TotalNnzMaxPerCCol", ["all", "densityMult", "maxNumPerCCol"]
)
# precomputed data
PreTotalNumNnz = co.namedtuple("PreTotalNumNnz", ["cCutMultiplyer"])
# common parameters of optimization
Params = co.namedtuple(
    "Params",
    [
        "model",
        "numIterations",
        "numColumnB",
        "numNz",
        "power",
        "alpha",
        "lr",
        "seed",
        "initB",
        "initC",
        "numBits",
    ],
)
# Result: dense matrices B and C
Res = co.namedtuple("Res", ["B", "C"])
