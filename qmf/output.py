# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

import igl
import numpy as np
import qmf.dataStructures as ds
import qmf.geometry as geometry
import rig.riglogic as rl
import termcolor as fn
import torch


def loadAnimWeights(p, npb):
    modelsWithRig = ("aura", "bowen", "jupiter", "proteus")
    if p.model not in modelsWithRig:
        npb = geometry.loadModel("jupiter")

    inbetween_dict = npb["inbetween_info"].item()
    corrective_dict = npb["combination_info"].item()

    test_anim = np.load(f"{geometry.dataPath}/test_anim.npz")
    # anim_weights num_frames x num_blendshapes
    # one weight per blendshape per frame
    return rl.compute_rig_logic(
        torch.from_numpy(test_anim["weights"][:, :72]).float(),
        inbetween_dict,
        corrective_dict,
    ).numpy()


def nameFromParam(p):
    if isinstance(p.numNz, ds.TotalNnzMaxPerCCol):
        densityMult = "_CM" if p.numNz.densityMult != 0.0 else ""
        return f"Nnz{p.numNz.all}{densityMult}_Nb{p.numColumnB}_a{int(p.alpha)}"
    elif isinstance(p.numNz, ds.ConstPerCCol):
        return f"NnzB{p.numNz.matrixB}_cC{p.numNz.columnC}_Nb{p.numColumnB}_a{int(p.alpha)}"
    elif isinstance(p.numNz, ds.ConstPerMatrix):
        return f"NnzB{p.numNz.matrixB}_NnzC{p.numNz.matrixC}_Nb{p.numColumnB}_a{int(p.alpha)}"
    else:
        return f"Nnz{p.numNz.all}_Nb{p.numColumnB}_a{int(p.alpha)}"


def folderNameFromParam(p):
    return f"out/{p.model}/{nameFromParam(p)}"


def outputResultsBlendshapesObj(p, geo, res, postfix=""):
    B, C = res
    faces = geo.faces
    restPos = geo.restPos * geo.scale
    BC = B @ C * geo.scale
    BCCpu = BC.detach().cpu().numpy()
    numBS = BCCpu.shape[0] // 3
    quantization = f"_Q{p.numBits}" if p.numBits != 0 else ""
    folderName = f"{folderNameFromParam(p)}/objBS{quantization}"
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    print(f"output Results BlendshapesObj -> {folderName}")
    for i in range(numBS):
        oneShape = BCCpu[i * 3 : (i + 1) * 3, :].transpose()
        igl.write_obj(f"{folderName}/BS{i:05d}.obj", restPos + oneShape, faces)
    # save rest pose
    igl.write_obj(f"{folderName}/BS{numBS:05d}.obj", restPos, faces)


def outputBlendshapesObj(model, geo):
    restPos = geo.restPos * geo.scale
    ACpu = geo.A.detach().cpu().numpy() * geo.scale
    numBS = ACpu.shape[0] // 3
    folderName = f"out/{model}/objBS_lossless"
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    print(f"output Blendshapes (obj) -> {folderName}")
    for i in range(numBS):
        oneShape = ACpu[i * 3 : (i + 1) * 3, :].transpose()
        igl.write_obj(
            f"{folderName}/{model}BS{i:05d}.obj", restPos + oneShape, geo.faces
        )
    # save rest pose
    igl.write_obj(f"{folderName}/{model}BS{numBS:05d}.obj", restPos, geo.faces)


def outputResultsObj(p, geo, res, frameRange=None, maxNumBS=None, postfix=""):
    def weight33(bsWeights):
        numBS = bsWeights.shape[0]
        res = bsWeights.reshape(1, 1, numBS) * np.dstack([np.eye(3)] * numBS)
        # ┌      ┐┌      ┐┌      ┐
        # │w₁   0││w₂   0││w₃   0│
        # │  w₁  ││  w₂  ││  w₃  │  ───▶ axis 2
        # │0   w₁││0   w₂││0   w₃│
        # └      ┘└      ┘└      ┘
        return res.transpose(0, 2, 1).reshape(3, -1)
        # ┌                  ┐
        # │w₁0 0 w₂0 0 w₃0 0 │
        # │0 w₁0 0 w₂0 0 w₃0 │
        # │0 0 w₁0 0 w₂0 0 w₃│
        # └                  ┘

    B, C = res
    animWeights = loadAnimWeights(p, geo.npb)
    BCpu = B.detach().cpu().numpy()
    CCpu = C.detach().cpu().numpy()

    numFrames = animWeights.shape[0]

    quantization = f"_Q{p.numBits}" if p.numBits != 0 else ""
    folderName = f"{folderNameFromParam(p)}/objAn{postfix}{quantization}"
    print(f"output test animation (obj) -> {folderName}")
    if not os.path.exists(folderName):
        os.makedirs(folderName)

    BCCpu = BCpu @ CCpu * geo.scale
    if maxNumBS:
        print(f"{fn.fgRed}outputResultsObj maxNumBS: {maxNumBS}{fn.reset}")

    frameRange = range(numFrames) if not frameRange else frameRange
    for i in frameRange:
        weights = animWeights[i, :][: geo.numBS]
        if maxNumBS:
            weights[maxNumBS:] = 0.0
        w33 = weight33(weights)

        wBC = w33 @ BCCpu
        igl.write_obj(
            f"{folderName}/{p.model}{i:05d}.obj",
            geo.restPos * geo.scale + wBC.transpose(),
            geo.faces,
        )
