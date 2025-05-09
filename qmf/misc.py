# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import qmf.metrics as mt
import termcolor as fn


def printResults(title, maxDelta, meanDelta, meanEdgeAngeDiff, res):
    resultNumNonZeros = res.C.count_nonzero() + res.B.count_nonzero()
    if title:
        print(f"{title:^56}")
    print(
        f"|{'maxD':^10}|{'meanD':^12}|{'meanEAngle':^12}|{'numNz':^10}|{'B':^10}|{'C':^10}|"
    )
    print(
        f"{fn.bgBlue}"
        f"|{maxDelta:<10.7}|{meanDelta:<12.7}"
        f"|{meanEdgeAngeDiff:<12.7}"
        f"|{resultNumNonZeros:<10_}|{res.B.count_nonzero():<10_}|{res.C.count_nonzero():<10_}|"
        f"{fn.reset}"
    )


def calcAndPrintMetrics(geo, res):
    maxDelta, meanDelta = mt.calculateDeltas(geo.A, res.B, res.C)

    meanEdgeAngleDiff = mt.calculateAverageEdgeAngularDiff(
        geo.A, res.B, res.C, geo.restPos, geo.triangles
    )

    printResults(
        "",
        maxDelta * geo.scale,
        meanDelta * geo.scale,
        meanEdgeAngleDiff,
        res,
    )
    return (
        maxDelta * geo.scale,
        meanDelta * geo.scale,
        meanEdgeAngleDiff,
    )


def houdiniIndicesToList(s):
    res = []
    for i in s.split(" "):
        p = i.split("-")
        if len(p) == 1:
            res.append(int(p[0]))
        else:
            res += list(range(int(p[0]), int(p[1]) + 1))
    return res
