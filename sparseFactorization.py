# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import qmf.bestSeed as bs
import qmf.compress as comp
import qmf.dataStructures as ds
import qmf.geometry as geometry
import qmf.init as init
import qmf.misc as mt
import qmf.output as out
import qmf.quantization as qn
import qmf.train as train
import torch

geometry.dataPath = "models/"


def runModel():
    """Optimize one or several models"""

    modelAlpas = {
        "aura": 9,
        "bowen": 35,
        "jupiter": 8,
        "proteus": 25,
    }
    paperParams = {
        "aura": ds.TotalNnzMaxPerCCol(
            all=35_000, densityMult=7.26e-4, maxNumPerCCol=30
        ),
        "bowen": ds.TotalNnzMaxPerCCol(
            all=110_000, densityMult=0.000684, maxNumPerCCol=30
        ),
        "jupiter": ds.TotalNnzMaxPerCCol(
            all=30_000, densityMult=0.000588, maxNumPerCCol=20
        ),
        "proteus": ds.TotalNnzMaxPerCCol(
            all=110_000, densityMult=0.00262, maxNumPerCCol=20
        ),
    }

    for model in ["aura"]:  # ["bowen", "jupiter", "proteus"]:
        # load geometry, calculate laplacian, curvetures, etc
        geo = geometry.calcGeo(model)

        p = ds.Params(
            model=model,
            numIterations=20_000,
            numColumnB=200,
            numNz=paperParams[model],
            power=2,
            alpha=modelAlpas[model],
            lr=1e-2,
            seed=1,
            initB=1e-3,
            initC=1e-3,
            numBits=0,
        )
        # precalculate data for culling algorithm
        pre = geometry.precomputeWrinklesDensity(p, geo)
        bestSeed = bs.chooseBestSeed(p, geo, pre, 1_000, 15)
        p = p._replace(seed=bestSeed)

        res = init.initRes(p, geo.numBS, geo.numVertices)
        optimizer = torch.optim.Adam([res.B, res.C], lr=p.lr, betas=(0.9, 0.9))

        print(p)

        res, _ = train.train(optimizer, p, geo, pre, res)

        mt.calcAndPrintMetrics(geo, res)

        compressedRes = comp.compressResults(p, res)
        comp.saveOptimizationResults(p, geo, res, compressedRes)
        out.outputResultsBlendshapesObj(p, geo, res)
        out.outputResultsObj(p, geo, res, frameRange=range(225))

        # save results with different quantization level (0 - float)
        for numBits in [0, 8, 11_11_10, 16]:
            p = p._replace(numBits=numBits)
            compressedRes = comp.compressResults(p, res)

            deQuantizedRes = ds.Res(
                B=qn.tensorQuantizationDeQuantization(res.B, numBits),
                C=qn.tensorQuantizationDeQuantization(res.C, numBits),
            )
            print(f"quantization {numBits} bits")
            mt.calcAndPrintMetrics(geo, deQuantizedRes)
            out.outputResultsBlendshapesObj(p, geo, deQuantizedRes)
            out.outputResultsObj(p, geo, deQuantizedRes, frameRange=range(225))

            comp.saveOptimizationResults(p, geo, res, compressedRes)


def main():
    runModel()
    # output lossless blendshapes for comparison
    # for model in ("aura",):
    #     outputBlendshapesObj(model, calcGeo(model))


if __name__ == "__main__":
    main()
