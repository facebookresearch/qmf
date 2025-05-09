# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import csv

from collections import OrderedDict
from pprint import pprint
from textwrap import dedent

# import prefixed as pf
import numpy as np

from math import *
import itertools as it

import matplotlib.pyplot as plt

models = ("aura", "jupiter", "boz", "proteus", "louise", "carlos")

fastQ = {
    "aura": 8,
    "boz": 16,
    "jupiter": 8,
    "proteus": 16,
    "louise": 16,
    "carlos": 16,
}

smallQ = {
    "aura": 8,
    "boz": 11_11_10,
    "jupiter": 8,
    "proteus": 11_11_10,
    "louise": 11_11_10,
    "carlos": 11_11_10,
}

skinningDecOrigDataFiles = {
    "boz": "boz/data_Nb400_Nw8_Nnz6000_a50.npz",
    "aura": "aura/data_Nb400_Nw8_Nnz6000_a20.npz",
    "jupiter": "jupiter/data_Nb400_Nw8_Nnz6000_a10.npz",
    "proteus": "proteus/data_Nb400_Nw8_Nnz6000_a50.npz",
    "louise": "louise/data_Nb400_Nw8_Nnz20000_a8.npz",
    "carlos": "carlos/data_Nb400_Nw8_Nnz20000_a10.npz",
}
skinningDecDataFiles = {
    "boz": "boz/data_bxToErr_Nb400_Nw8_Nnz6000_a50.npz",
    "aura": "aura/data_bxToErr_Nb400_Nw8_Nnz6000_a20.npz",
    "jupiter": "jupiter/data_bxToErr_Nb400_Nw8_Nnz6000_a10.npz",
    "proteus": "proteus/data_bxToErr_Nb400_Nw8_Nnz6000_a50.npz",
    "louise": "louise/data_bxToErr_Nb400_Nw8_Nnz20000_a8.npz",
    "carlos": "carlos/data_bxToErr_Nb400_Nw8_Nnz20000_a10.npz",
}


# # output Results BlendshapesObj -> out/dragon/objBS_Nnz300000_Nb200_a8_Q8
# # saveOptimizationResults -> out/dragon/data_Nnz300000_Nb200_a8_Q8.npz
# # saveOptimizationResults -> out/dragon/data_Nnz300000_Nb200_a8.npz
# sparseFactorizationQ8DataFiles = {
#     # "aura": "aura/aura/Nnz35000_CM_Nb200_a25/data_Q8.npz",
#     "aura": "aura/data_Nnz35000_CM_Nb200_a8_Q16.npz",
#     "boz": "boz/data_Nnz110000_CM_Nb200_a25_Q16.npz",
#     "jupiter": "jupiter/data_Nnz30000_CM_Nb200_a8_Q16.npz",
#     "proteus": "proteus/data_Nnz110000_CM_Nb200_a20_Q16.npz",
#     "louise": "louise/data_Nnz130000_CM_Nb200_a8_Q16.npz",
#     "carlos": "carlos/data_Nnz140000_CM_Nb200_a8_Q16.npz",
#     # "dragon": "dragon/data_Nnz300000_Nb200_a8_Q16.npz",
# }
# sparseFactorizationFloatDataFiles = {
#     "aura": "aura/data_Nnz35000_CM_Nb200_a8.npz",
#     "boz": "boz/data_Nnz110000_CM_Nb200_a25.npz",
#     "jupiter": "jupiter/data_Nnz30000_CM_Nb200_a8.npz",
#     "proteus": "proteus/data_Nnz110000_CM_Nb200_a20.npz",
#     "louise": "louise/data_Nnz130000_CM_Nb200_a8.npz",
#     "carlos": "carlos/data_Nnz140000_CM_Nb200_a8.npz",
#     # "dragon": "dragon/data_Nnz300000_Nb200_a8.npz",
# }
qmfFloatDataFiles = {
    "aura": "aura/Nnz35000_CM_Nb200_a9/data.npz",
    "boz": "boz/Nnz110000_CM_Nb200_a35/data.npz",
    "jupiter": "jupiter/Nnz30000_CM_Nb200_a8/data.npz",
    "proteus": "proteus/Nnz110000_CM_Nb200_a25/data.npz",
    "louise": "louise/Nnz130000_CM_Nb200_a26/data.npz",
    "carlos": "carlos/Nnz140000_CM_Nb200_a25/data.npz",
}

qmfFastDataFiles = {
    "aura": "aura/Nnz35000_CM_Nb200_a9/data_Q8.npz",
    "boz": "boz/Nnz110000_CM_Nb200_a35/data_Q16.npz",
    "jupiter": "jupiter/Nnz30000_CM_Nb200_a8/data_Q8.npz",
    "proteus": "proteus/Nnz110000_CM_Nb200_a25/data_Q16.npz",
    "louise": "louise/Nnz130000_CM_Nb200_a26/data_Q16.npz",
    "carlos": "carlos/Nnz140000_CM_Nb200_a25/data_Q16.npz",
}
qmfSmallDataFiles = {
    "aura": "aura/Nnz35000_CM_Nb200_a9/data_Q8.npz",
    "boz": "boz/Nnz110000_CM_Nb200_a35/data_Q111110.npz",
    "jupiter": "jupiter/Nnz30000_CM_Nb200_a8/data_Q8.npz",
    "proteus": "proteus/Nnz110000_CM_Nb200_a25/data_Q111110.npz",
    "louise": "louise/Nnz130000_CM_Nb200_a26/data_Q111110.npz",
    "carlos": "carlos/Nnz140000_CM_Nb200_a25/data_Q111110.npz",
}

sparseFactFolder = (
    "/home/romanfedotov/Documents/prototypes/sparseFactorizationNext/out/"
)
skinningDecFolder = (
    "/home/romanfedotov/Documents/prototypes/blendshapesCompression/out/"
)


floatSize = 4
float16Size = 2
vecSize = 3 * floatSize
uint32Size = 4
uint16Size = 2
uint8Size = 1


def printTime(t):
    unitSymbol = ["$\\mu$s", "ms", "s"]
    # unitSymbol = ["us", "ms", "s"]
    i = floor(log(t) / log(1000.0))
    if i > 0 or t > 0.1:
        raise Exception("time value outside of printTime function scope")
    return f"{t / (1000.0**i):.1f} {unitSymbol[i - 1]}"


def printSize(s):
    if s == 0:
        return "0 B"
    unitSymbol = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = floor(log(s) / log(1024))
    return f"{s / (1024**i):.1f} {unitSymbol[i]}"


def readTable(device):
    def cleanName(name):
        return (
            name.split("/")[0]
            .replace("BM_cuda10", "")
            .replace("BM_cpu", "")
            .replace('"', "")
        )

    csvFileName = f"input/{device}_results.csv"
    with open(csvFileName, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        headerSize = 10
        table = [row for row in spamreader][headerSize:]
        header = table[0]

        d = {
            cleanName(i[0]): {k: v for k, v in zip(header[1:], i[1:])}
            for i in table[1:]
        }
    return d


tables = {"CPU": readTable("CPU"), "CUDA": readTable("CUDA")}


def getTableValue(table, modelName, algorithm, quantization, columnName):
    algorithmName = {"sbs": "Sbs", "qmf": "QMF", "csfb": "SkinDecomp"}
    quantizationName = {8: "uint8_t", 11_11_10: "uint32_t", 16: "uint16_t", 0: "float"}
    if algorithm == "qmf":
        key = f"QMF_{modelName}<{quantizationName[quantization]}>"
    else:
        key = f"{algorithmName[algorithm]}_{modelName}"

    return table[key][columnName]


def getDuration(device, modelName, algorithm, quantization):
    return (
        float(
            getTableValue(
                tables[device], modelName, algorithm, quantization, "real_time"
            )
        )
        * 1e-9
    )


# def getCUDADuration(modelName, algorithm, quantization):
#     return float(getTableValue(tableCUDA, modelName, algorithm, quantization, "real_time")) * 1e-9


def getRuntimeSize(device, modelName, algorithm, quantization):
    return int(
        getTableValue(
            tables[device], modelName, algorithm, quantization, '"TotalBytes"'
        )
    )


def getRuntimeScratchSize(device, modelName, algorithm, quantization):
    return int(
        getTableValue(
            tables[device], modelName, algorithm, quantization, '"PerInstanceScratch"'
        )
    )


# def getCUDARuntimeSize(modelName, algorithm, quantization):
#     return int(getTableValue(tableCUDA, modelName, algorithm, quantization, '"TotalBytes"'))


def getSparseBSSize(modelName):
    path = "/home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/out/"
    res = np.load(f"{path}{modelName}_SBData.npz")
    return res["numSparseDeltas"] * (vecSize + uint32Size)


def getNumVerticesNumBS(modelName):
    path = "/home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/out/"
    res = np.load(f"{path}{modelName}_SBData.npz")
    return res["numVertices"], res["numBlendshapes"]


def getDenseBSSize(modelName):
    numVertices, numBlendshapes = getNumVerticesNumBS(modelName)
    return numBlendshapes * numVertices * vecSize


def getSkinningDecompositionSizeF(fileName):
    results = np.load(fileName, allow_pickle=True)
    numNzBrt = np.count_nonzero(results["Brt"])
    numIndices = results["skinIndices"].size
    numWeights = results["skinWeights"].size

    return (
        numNzBrt * (floatSize + uint32Size)
        + numIndices * uint16Size
        + numWeights * floatSize
    )


def getMaxMeanDeltasF(fileName):
    results = np.load(fileName, allow_pickle=True)["deltas"].tolist()
    if len(results) < 3:
        results.append(0.0)
    return results


def getSparseFactorizationSizeF(fileName):
    res = np.load(fileName, allow_pickle=True)
    arrayNames = (
        "BData",
        "BColIndices",
        "BNumPerRow",
        "quantizationParamsB",
        "CData",
        "CRowIndices",
        "CNumPerCol",
        "quantizationParamsC",
    )
    return sum([res[n].nbytes for n in arrayNames], 0)


def getSkinningDecompositionSize(modelName):
    fileNameSD = f"{skinningDecFolder}{skinningDecDataFiles[modelName]}"
    return getSkinningDecompositionSizeF(fileNameSD)


def getQMFFloatSize(modelName):
    fileNameSF = f"{sparseFactFolder}{qmfFloatDataFiles[modelName]}"
    return getSparseFactorizationSizeF(fileNameSF)


def getQMFFastSize(modelName):
    fileNameSF = f"{sparseFactFolder}{qmfFastDataFiles[modelName]}"
    return getSparseFactorizationSizeF(fileNameSF)


def getQMFSmallSize(modelName):
    fileNameSF = f"{sparseFactFolder}{qmfSmallDataFiles[modelName]}"
    return getSparseFactorizationSizeF(fileNameSF)


# -----------------------------------overheads------------------------------------
def getCudaOverhead(modelName, algorithm):
    if algorithm == "sbs":
        gpuPerModelBytes = float(
            getTableValue(tables["CUDA"], modelName, "sbs", 0, '"PerModelBytes"')
        )
        cpuPerModelBytes = float(
            getTableValue(tables["CPU"], modelName, "sbs", 0, '"PerModelBytes"')
        )
        return int(gpuPerModelBytes - cpuPerModelBytes)
    if algorithm == "csfb":
        return getRuntimeScratchSize("CUDA", modelName, "csfb", 0)
    if algorithm == "qmf":
        gpuSize = getRuntimeSize("CUDA", modelName, "qmf", 0)
        cpuSize = getRuntimeSize("CPU", modelName, "qmf", 0)
        scratchSize = getRuntimeScratchSize("CPU", modelName, "qmf", 0)
        return gpuSize - cpuSize + scratchSize


def getCpuOverhead(modelName, algorithm):
    if algorithm == "sbs":
        return 0
    if algorithm == "csfb":
        return getRuntimeScratchSize("CUDA", modelName, "csfb", 0)
    if algorithm == "qmf":
        return getRuntimeScratchSize("CPU", modelName, "qmf", 0)


# s = getRuntimeSize("CUDA", "aura", "qmf", 0)
# print(printSize(s))
# s = getRuntimeSize("CUDA", "aura", "qmf", 8)
# print(printSize(s))
#
# s = getRuntimeSize("CPU", "aura", "qmf", 0)
# print(printSize(s))
# s = getRuntimeSize("CPU", "aura", "qmf", 8)
# print(printSize(s))
#
#
# s = getQMFFloatSize("aura") + 2400
# print(printSize(s))
# s = getQMFSmallSize("aura") + 2400
# print(printSize(s))
#
# auraGOver = getRuntimeSize("CUDA", "aura", "qmf", 0) - getQMFFloatSize("aura")
# print(f"auraGOver {printSize(auraGOver)}")
# auraCOver = getRuntimeSize("CPU", "aura", "qmf", 0) - getQMFFloatSize("aura")
# print(f"auraCOver {printSize(auraCOver)}")
#
# bozGOver = getRuntimeSize("CUDA", "boz", "qmf", 0) - getQMFFloatSize("boz")
# print(f"bozGOver {printSize(bozGOver)}")
# bozCOver = getRuntimeSize("CPU", "boz", "qmf", 0) - getQMFFloatSize("boz")
# print(f"bozCOver {printSize(bozCOver)}")

# ------------------------------end of overhead calc------------------------------


def getMaxMeanDeltas(modelName, algorithm):
    if algorithm == "sd0":
        return getMaxMeanDeltasF(
            skinningDecFolder + skinningDecOrigDataFiles[modelName]
        )
    elif algorithm == "sd":
        return getMaxMeanDeltasF(skinningDecFolder + skinningDecDataFiles[modelName])
    elif algorithm == "qmf_float":
        return getMaxMeanDeltasF(sparseFactFolder + qmfFloatDataFiles[modelName])
    elif algorithm == "qmf_fast":
        return getMaxMeanDeltasF(sparseFactFolder + qmfFastDataFiles[modelName])
    elif algorithm == "qmf_small":
        return getMaxMeanDeltasF(sparseFactFolder + qmfSmallDataFiles[modelName])
    else:
        raise Exception(f"Can't find algorithm {algorithm}")


# a = getMaxMeanDeltas("aura", "sd")
# print(f"a {a}")
# a = getMaxMeanDeltas("aura", "sd0")
# print(f"a {a}")
# a = getMaxMeanDeltas("louise", "qmf_fast")
# print(f"a {a}")

#    maxD   |   meanD    | meanAngle  | meanEAngle |  numNz   |    B     |    C
# 0.3861793 |0.003299295 |0.008099074 |0.004839506 |35_000    |4_555     |30_445

# --------------------------------------------------------------------------------
#                                      LaTeX
# --------------------------------------------------------------------------------


def buildModelsAlgorithmsDigaram(
    models,
    colors,
    algorithms,
    modelValues,
    yLabel,
    title,
    fileName=None,
    yLim=None,
    legendPos=None,
):
    algorithmSizes = dict(zip(algorithms, zip(*modelValues)))

    fig, ax = plt.subplots(layout="constrained", figsize=(6.4, 4.0))  # 6.4 4.8 # , )
    sizes = fig.get_size_inches()
    print(f"sizes {sizes}")

    x = np.arange(len(modelValues))  # the label locations
    width = 1.0 / (len(algorithms) + 0.5)  # the width of the bars
    for (model, sizes), multiplier in zip(algorithmSizes.items(), it.count()):
        offset = width * multiplier
        rects = ax.bar(x + offset, sizes, width, label=model, color=colors[multiplier])
        # ax.bar_label(rects, padding=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(yLabel)
    if yLim:
        ax.set_ylim(yLim)
    # ax.set_title(title)
    ax.set_xticks(x + width, models)
    ax.set_axisbelow(True)
    # ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.grid(axis="y")
    if legendPos:
        ax.legend(loc=legendPos)  # , ncols=3
    else:
        ax.legend(loc="upper left")  # , ncols=3
    # ax.set_ylim(0, 250)
    if fileName:
        plt.savefig(fileName, format="pdf", bbox_inches="tight")
    plt.show()


def buildSizes(fileName):
    # models = list(qmfFloatDataFiles.keys())
    modelSizes = [
        (
            # getSparseBSSize(model) ,
            getSparseBSSize(model) / getSkinningDecompositionSize(model),
            getSparseBSSize(model) / getQMFFloatSize(model),
            getSparseBSSize(model) / getQMFSmallSize(model),
            getSparseBSSize(model) / getQMFFastSize(model),
        )
        for model in models  # runtimeFilesFloat.keys()
    ]
    algorithms = ("CSFB", "MF", "QMFSmall", "QMFFast")  # "sparse BS",
    printModelName = list(models)
    printModelName[2] = "bowen"
    buildModelsAlgorithmsDigaram(
        printModelName,
        # ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:olive"],
        ["#006c8a", "#009f86", "#60ca54", "#ffe30f"],
        algorithms,
        modelSizes,
        "compression factor (x)",
        "Compression rate relative to sparse blendshapes",
        fileName,
        legendPos="upper right",
    )


def buildDuration(device, fileName):
    # cpuSB, cpuSD, cpuSF_float = getDurationFloat(modelName)
    # _, _, cpuSF_Q8 = getDurationQ8(modelName)

    # models = list(qmfFloatDataFiles.keys())
    k = 1_000_000
    modelCpu = [
        # np.array(getDurationFloat(model) + [getDurationQ8(model)[-1]]) / k
        np.array(
            [
                getDuration(device, model, "sbs", 0),
                getDuration(device, model, "csfb", 0),
                getDuration(device, model, "qmf", 0),
                getDuration(device, model, "qmf", smallQ[model]),
                getDuration(device, model, "qmf", fastQ[model]),
            ]
        )
        * k
        for model in models  # runtimeFilesFloat.keys()
    ]
    algorithms = ("sparse BS", "CSFB", "MF", "QMFSmall", "QMFFast")  # "sparse BS",
    printModelName = list(models)
    printModelName[2] = "bowen"
    buildModelsAlgorithmsDigaram(
        printModelName,
        # ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:olive"],
        ["#043a5c", "#006c8a", "#009f86", "#60ca54", "#ffe30f"],
        algorithms,
        modelCpu,
        "time (microseconds)",
        "Average facial expression calculation time",
        fileName,
        [0, 1200] if device == "CPU" else [0, 1500],
        None if device == "CPU" else "upper right",
    )


def buildModelsTable(fileName):
    # def tableSection(text, numModels):
    #     return r"\multicolumn{" + str(numModels) + r"}{c}{\textbf{" + text + "}}"

    tableBegin = dedent(
        r"""
        \begin{table}[t]
        \begin{center}
        \begin{tabular}{ l r r r r }
          \hline
          model & num Vertices & num BS & Dens BS & Sparse BS\\
          \hline """
    )

    tableEnd = dedent(
        r"""
        \hline
    \end{tabular}
    \end{center}
    \caption{Test Models}
    \label{tab:TestModels}
    \end{table} """
    )

    with open(fileName, "w") as f:
        f.write(tableBegin)
        for modelName in models:
            numVertices, numBS = getNumVerticesNumBS(modelName)
            denseBSSize = printSize(getDenseBSSize(modelName))
            sparseBSSize = printSize(getSparseBSSize(modelName))
            f.write(
                f"  {modelName} & {numVertices} & {numBS} & {denseBSSize} & {sparseBSSize} \\\\\n"
            )
        f.write(tableEnd)


def buildBigTable(fileName):
    def tableSection(text, numModels):
        return r"\multicolumn{" + str(numModels) + r"}{c}{\textbf{" + text + "}} \\\\"

    # fmt: off
    tableBegin = dedent( r"""
        \begin{table*}[t]
        \begin{center}
        """[1:])

    tableEnd = dedent( r"""
          \hline
          \end{tabular}
        \end{center}
        \caption{Results. Note that GPU time is for animating 10 faces concurrently.}
        \label{tab:FullResults}
        \end{table*}
        """[1:])
    # fmt: on

    numModels = len(models)
    tableHeader = r"\begin{tabular}{ l " + " r" * numModels + " }"

    class LaTaXTable:
        def __init__(self):
            self.table = []
            self.i = it.count(0)

        def newColumn(self):
            self.i = it.count(0)

        def add(self, key, value):
            i = next(self.i)
            if i >= len(self.table):
                self.table.append(key + " & " + str(value) + "\\\\")
            else:
                self.table[i] = self.table[i][:-2] + " & " + str(value) + "\\\\"

        def setRow(self, value):
            i = next(self.i)
            if i >= len(self.table):
                self.table.append(str(value))

    tbl = LaTaXTable()

    for modelName in models:
        printModelName = "bowen" if modelName == "boz" else modelName
        tbl.newColumn()

        tbl.setRow("\\hline")
        tbl.add(" ", printModelName.title())
        tbl.setRow("\\hline")
        numVertices, numBS = getNumVerticesNumBS(modelName)
        tbl.add("num vertices", numVertices)
        tbl.add("num BS", numBS)
        tbl.add("dense BS", printSize(getDenseBSSize(modelName)))

        tbl.setRow(tableSection("Sparse Blendshapes", numModels))
        tbl.add("size", printSize(getSparseBSSize(modelName)))
        tbl.add("CPU overhead", printSize(getCpuOverhead(modelName, "sbs")))
        tbl.add("GPU overhead", printSize(getCudaOverhead(modelName, "sbs")))
        tbl.add("CPU time", printTime(getDuration("CPU", modelName, "sbs", 0)))
        tbl.add("GPU time (10)", printTime(getDuration("CUDA", modelName, "sbs", 0)))

        tbl.setRow(tableSection("Skinning Decomposition (\\csfb)", numModels))
        maxD, meanD, meanEdgeAngleDiff = getMaxMeanDeltas(modelName, "sd0")
        tbl.add("MXE", f"{maxD:7.3} cm")
        tbl.add("MAE", f"{meanD:7.3} cm")
        tbl.add("EAD", f"{meanEdgeAngleDiff:7.3} rad")

        tbl.setRow(
            tableSection("Skinning Decomposition (\\csfb) + new loss", numModels)
        )
        tbl.add("size", printSize(getSkinningDecompositionSize(modelName)))
        tbl.add("CPU overhead", printSize(getCpuOverhead(modelName, "csfb")))
        tbl.add("GPU overhead", printSize(getCudaOverhead(modelName, "csfb")))
        tbl.add("CPU time", printTime(getDuration("CPU", modelName, "csfb", 0)))
        tbl.add("GPU time (10)", printTime(getDuration("CUDA", modelName, "csfb", 0)))
        maxD, meanD, meanEdgeAngleDiff = getMaxMeanDeltas(modelName, "sd")
        tbl.add("MXE", f"{maxD:7.3} cm")
        tbl.add("MAE", f"{meanD:7.3} cm")
        tbl.add("EAD", f"{meanEdgeAngleDiff:7.3} rad")

        tbl.setRow(tableSection("Sparse Factorization (\\mf)", numModels))
        tbl.add("size", printSize(getQMFFloatSize(modelName)))
        tbl.add("CPU overhead", printSize(getCpuOverhead(modelName, "qmf")))
        tbl.add("GPU overhead", printSize(getCudaOverhead(modelName, "qmf")))
        tbl.add("CPU time", printTime(getDuration("CPU", modelName, "qmf", 0)))
        tbl.add("GPU time (10)", printTime(getDuration("CUDA", modelName, "qmf", 0)))
        maxD, meanD, meanEdgeAngleDiff = getMaxMeanDeltas(modelName, "qmf_float")
        tbl.add("MXE", f"{maxD:7.3} cm")
        tbl.add("MAE", f"{meanD:7.3} cm")
        tbl.add("EAD", f"{meanEdgeAngleDiff:7.3} rad")

        tbl.setRow(
            tableSection("Quantized Sparse Factorization  (\\qmf) small", numModels)
        )
        tbl.add("size", printSize(getQMFSmallSize(modelName)))
        tbl.add(
            "CPU time",
            printTime(getDuration("CPU", modelName, "qmf", smallQ[modelName])),
        )
        tbl.add(
            "GPU time (10)",
            printTime(getDuration("CUDA", modelName, "qmf", smallQ[modelName])),
        )
        maxD, meanD, meanEdgeAngleDiff = getMaxMeanDeltas(modelName, "qmf_small")
        tbl.add("MXE", f"{maxD:7.3} cm")
        tbl.add("MAE", f"{meanD:7.3} cm")
        tbl.add("EAD", f"{meanEdgeAngleDiff:7.3} rad")

        tbl.setRow(
            tableSection("Quantized Sparse Factorization  (\\qmf) fast", numModels)
        )
        tbl.add("size", printSize(getQMFFastSize(modelName)))
        tbl.add(
            "CPU time",
            printTime(getDuration("CPU", modelName, "qmf", fastQ[modelName])),
        )
        tbl.add(
            "GPU time (10)",
            printTime(getDuration("CUDA", modelName, "qmf", fastQ[modelName])),
        )
        maxD, meanD, meanEdgeAngleDiff = getMaxMeanDeltas(modelName, "qmf_fast")
        tbl.add("MXE", f"{maxD:7.3} cm")
        tbl.add("MAE", f"{meanD:7.3} cm")
        tbl.add("EAD", f"{meanEdgeAngleDiff:7.3} rad")

    lines = [tableBegin, tableHeader] + tbl.table + [tableEnd]
    with open(fileName, "w") as f:
        f.write("\n".join(lines))


# generatedPath = "/home/romanfedotov/Documents/paperSF/latex/overleafHPG/generated/"
generatedPath = "temp/"


buildBigTable(generatedPath + "bigTable.tex")
# buildModelsTable(generatedPath + "modelsTable.tex")
# buildSizes(generatedPath + "compressionRatesDiagram.pdf")
# buildDuration("CPU", generatedPath + "durationDiagramCPU.pdf")
# buildDuration("CUDA", generatedPath + "durationDiagramCUDA.pdf")
