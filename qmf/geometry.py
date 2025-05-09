# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import collections as co

import igl
import numpy as np
import qmf.dataStructures as ds
import scipy as sp
import termcolor as fn
from qmf.numpyTorch import toNumpy, toTorch

dataPath = ""

Geo = co.namedtuple(
    "Geo",
    [
        "npb",
        "faces",
        "restPos",
        "bsDeltas",
        "scale",
        "numBS",
        "numVertices",
        "triangles",
        "A",
        "laplacian",
        "restCurvature",
        "curvature",
    ],
)


def loadModel(modelName):
    return np.load(f"{dataPath}/{modelName}.npz", allow_pickle=True)


def triangulateQuads(faces):
    """Function converts quads or triangles to triangles

    Args:
        faces (numpy.ndarray): Nf x 4 or Nf x 3 array of vertex indices

    Returns:
        2*Nf x 3 or Nf x 3 array of vertex indices
    """
    if faces.shape[1] == 4:
        return np.concatenate([faces[:, (0, 1, 2)], faces[:, (2, 3, 0)]], axis=0)
    if faces.shape[1] == 3:
        return faces


def calculateLaplacianRegularization(positions, triangles, faces):
    """function calculate graph Laplacian

    Args:
        positions (numpy.ndarray): Nv x 3 array of positions
        triangles (numpy.ndarray): Nf x 3 array of faces
        faces (numpy.ndarray): Nf x 4 or Nf x 3  array of quads or triangles

    Returns:
        graph Laplacian matrix
    """
    adj = igl.adjacency_matrix(faces)
    adj_diag = np.array(np.sum(adj, axis=1)).squeeze()
    # Rigidness Laplacian regularization. from "Robust and Accurate Skeletal Rigging from Mesh Sequences":
    # ⎧-1        if k = i
    # ⎨1/|N(i)|, if k ∈N(i)
    # ⎩0         otherwise.
    # Where N(i) denotes all the 1-ring neighbours of i
    Lg = sp.sparse.diags(1 / adj_diag) @ (adj - sp.sparse.diags(adj_diag))
    return toTorch(Lg.astype(np.float32).todense()).to_sparse()


def calcSignedMeanCurvature(positions, triangles):
    """function calculates sign curvature

    Args:
        positions (numpy.ndarray): Nv x 3 array of positions
        triangles (numpy.ndarray): Nv x 3 array of triangles

    Returns:
        array of per vertex signed curvatures (convex - positive, concave - negative)
    """
    cotLaplacian = igl.cotmatrix(positions, triangles)
    areas = igl.massmatrix(positions, triangles, igl.MASSMATRIX_TYPE_VORONOI)

    np.reciprocal(areas.data, out=areas.data)
    invAreas = areas
    # Laplace-Beltrami of position
    HN = -invAreas @ (cotLaplacian @ positions)
    H = np.linalg.norm(HN, axis=1)
    N = igl.per_vertex_normals(positions, triangles)
    Hsigns = np.sum(HN * N, axis=1)
    H = np.copysign(H, Hsigns)

    return H


def calcBSCurvature(restPos, bsDeltas, triangles):
    """function calculates curvature for every vertex in every blendshape

    Args:
        restPos (numpy.ndarray): Nv x 3 array of rest pose vertex positions
        bsDeltas (numpy.ndarray): Nbs x Nv x 3 array of blendshapes deltas
        triangles (numpy.ndarray): Nf x 3 array of rest pose triangles

    Returns:
        array of curvatures per vertex per blendshape
    """
    numBS = bsDeltas.shape[0]
    numVertices = restPos.shape[0]

    positionsBS = np.vstack([restPos] * numBS)
    deltas = bsDeltas.transpose(0, 1, 2).reshape(-1, 3)
    positionsBS += deltas

    offsets = np.arange(0, numVertices * numBS, numVertices)
    trianglesBS = np.dstack([triangles] * numBS) + offsets
    trianglesBS = trianglesBS.transpose(2, 0, 1).reshape(-1, 3)

    curvatureBS = calcSignedMeanCurvature(positionsBS, trianglesBS)

    # curvature.shape (numBS*NumVertices,) -> (numBs, numVertices)
    curvatureBS = curvatureBS.reshape(-1, numVertices)
    # TODO: Calculate Gaussian curvature

    return toTorch(curvatureBS)


def calcRestCurvature(restPos, triangles):
    """function calculates rest pose curvature

    Args:
        restPos (numpy.ndarray): Nv x 3 array of positions
        triangles (numpy.ndarray): Nf x 3 array of faces (indices to positions)

    Returns:
         array of per vertex curvature
    """
    return toTorch(calcSignedMeanCurvature(restPos, triangles))


def calcWrinklesDensity(geo):
    """function calculates wrinkles density map

    Args:
        geo (Geo): geometry data

    Returns:
        wrinkles density map
    """
    numTop = 5

    curvChange = toNumpy(geo.curvature) - toNumpy(geo.restCurvature)

    curvChangeAbs = np.abs(curvChange)
    maxAbsChange = np.max(curvChangeAbs, axis=0)

    part = np.partition(curvChangeAbs, -numTop, axis=0)
    lowMean = np.mean(part[:-numTop], axis=0)

    wrinklesDensity = maxAbsChange / lowMean

    return toTorch(wrinklesDensity)


def calcGeo(model, maxNumBS=0):
    """function loads and precalculates all geometry related data

    Args:
        model (str): name of the model
        maxNumBS (int): maximum number of blendshapes

    Returns:
        named tuple Geo
    """
    npb = loadModel(model)
    numBS = npb["deltas"].shape[0]

    data_matrix = npb["deltas"].transpose(1, 0, 2).reshape(-1, numBS * 3).transpose()

    bsDeltas = npb["deltas"]
    restPos = npb["rest_verts"]

    # scale = (restPos.max(axis=0) - restPos.min(axis=0)).max()
    # scale = np.absolute(bsDeltas).max()
    scale = 1.0
    print(f"geometry.scale {scale}")
    data_matrix *= 1.0 / scale
    bsDeltas *= 1.0 / scale
    restPos *= 1.0 / scale

    faces = npb["rest_faces"]
    triangles = triangulateQuads(faces)
    curvature = calcBSCurvature(restPos, bsDeltas, triangles)
    laplacian = calculateLaplacianRegularization(restPos, triangles, faces)

    if maxNumBS != 0 and numBS > maxNumBS:
        print(
            f"{fn.fgRed}numBS: {maxNumBS} (cut {numBS - maxNumBS} blendshapes){fn.reset}"
        )
        data_matrix = data_matrix[: maxNumBS * 3, :]
        numBS = maxNumBS
        curvature = curvature[:maxNumBS, :]

    geo = Geo(
        # A is blendshapes deltas matrix. (goal of optimization)
        # shape: (num_blendShapes * 3 (x, y, z), num vertices)
        npb=npb,
        faces=faces,
        restPos=restPos,
        bsDeltas=bsDeltas,
        scale=scale,
        A=toTorch(data_matrix),
        numBS=numBS,
        numVertices=data_matrix.shape[1],
        triangles=triangles,
        laplacian=laplacian,  # cotanLaplacian,
        restCurvature=calcRestCurvature(restPos, triangles),
        curvature=curvature,
    )
    return geo


def precomputeWrinklesDensity(p, geo):
    """function precompute wrinkles density map

    Args:
        p (Param): parameters of optimization
        geo (Geo): Geometry structure

    Returns:
        return wrinkles density map
    """
    cMult = 1.0
    if p.numNz.densityMult > 0.0:
        wrinklesDensity = calcWrinklesDensity(geo)

        cMult = wrinklesDensity * p.numNz.densityMult + 1.0

        print(
            f"{fn.fgCyan}cMult min {cMult.min():.6} {fn.fgBlue}mean {cMult.mean():.6} {fn.fgCyan}max {cMult.max():.6}{fn.reset}"
        )
    return ds.PreTotalNumNnz(cCutMultiplyer=cMult)
