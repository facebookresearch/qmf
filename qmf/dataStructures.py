# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


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
