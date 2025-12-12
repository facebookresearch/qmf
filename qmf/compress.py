# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import collections
import os

import numpy as np
import qmf.dataStructures as ds
import qmf.metrics as mt
import qmf.output as out
import qmf.quantization as quant
import scipy as sp
from qmf.numpyTorch import toNumpy, toTorch

# Compressed results: B in CSR C in CSC, quantized values
CompressedRes = collections.namedtuple(
    "CompressedRes",
    [
        "BData",
        "BColIndices",
        "BNumPerRow",
        "quantizationParamsB",
        "CData",
        "CRowIndices",
        "CNumPerCol",
        "quantizationParamsC",
    ],
)


def compressResults(p, res):
    bitsToType = {0: np.float32, 8: np.uint8, 16: np.uint16, 11_11_10: np.uint32}
    dataType = bitsToType[p.numBits]
    indexType = np.uint8

    csrMatrixB = sp.sparse.csr_matrix(toNumpy(res.B))
    quantizedBData, bMin, bRange = quant.tensorQuantizationAndPack(
        toTorch(csrMatrixB.data), p.numBits
    )
    quantizedBData = toNumpy(quantizedBData).astype(dataType)
    quantizedBIndices = csrMatrixB.indices.astype(indexType)
    quantizedBNumPerRow = (csrMatrixB.indptr[1:] - csrMatrixB.indptr[:-1]).astype(
        indexType
    )

    csrMatrixC = sp.sparse.csc_matrix(toNumpy(res.C))
    quantizedCData, cMin, cRange = quant.tensorQuantizationAndPack(
        toTorch(csrMatrixC.data), p.numBits
    )
    quantizedCData = toNumpy(quantizedCData).astype(dataType)
    quantizedCRowIndices = csrMatrixC.indices.astype(indexType)
    quantizedCNumPerColumn = (csrMatrixC.indptr[1:] - csrMatrixC.indptr[:-1]).astype(
        indexType
    )

    return CompressedRes(
        BData=quantizedBData,
        BColIndices=quantizedBIndices,
        BNumPerRow=quantizedBNumPerRow,
        quantizationParamsB=np.array([float(bMin), float(bRange)]),
        CData=quantizedCData,
        CRowIndices=quantizedCRowIndices,
        CNumPerCol=quantizedCNumPerColumn,
        quantizationParamsC=np.array([float(cMin), float(cRange)]),
    )


def saveOptimizationResults(p, geo, res, compressedRes):
    numBits = 11 if p.numBits == 11_11_10 else p.numBits
    quantizedRes = ds.Res(
        B=quant.tensorQuantizationDeQuantization(res.B, numBits),
        C=quant.tensorQuantizationDeQuantization(res.C, numBits),
    )
    maxDelta, meanDelta = mt.calculateDeltas(geo.A, quantizedRes.B, quantizedRes.C)
    meanEdgeAngleDiff = mt.calculateAverageEdgeAngularDiff(
        geo.A, quantizedRes.B, quantizedRes.C, geo.restPos, geo.triangles
    )

    folderName = out.folderNameFromParam(p)
    if not os.path.exists(folderName):
        os.makedirs(folderName)

    quantization = f"_Q{p.numBits}" if p.numBits != 0 else ""
    fileName = f"{folderName}/data{quantization}.npz"
    print(f"save result of optimization -> {fileName}")

    np.savez(
        fileName,
        # float matrices
        matrixC=toNumpy(res.C),
        matrixB=toNumpy(res.B),
        # float deltas
        deltas=np.array([maxDelta, meanDelta, float(meanEdgeAngleDiff)]),
        # quantized B
        BData=compressedRes.BData,
        BColIndices=compressedRes.BColIndices,
        BNumPerRow=compressedRes.BNumPerRow,
        quantizationParamsB=compressedRes.quantizationParamsB,
        # quantized C
        CData=compressedRes.CData,
        CRowIndices=compressedRes.CRowIndices,
        CNumPerCol=compressedRes.CNumPerCol,
        quantizationParamsC=compressedRes.quantizationParamsC,
    )
