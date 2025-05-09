# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import igl
import numpy as np
import torch
from qmf.numpyTorch import toNumpy, toTorchCpu


def calculateDeltas(A, B, C):
    origDeltas = toNumpy(A)
    ourDeltas = toNumpy(B @ C)

    absDiff = np.abs(origDeltas - ourDeltas)

    maxDelta = absDiff.max()
    meanDelta = absDiff.mean()

    return maxDelta, meanDelta


def calcNormals(v, f):
    num_triangles = f.shape[0]
    faces_view = f.view(-1)
    triangle_vertices = v[faces_view, :]  # build from value and indices  shape(F*3, 3)
    triangle_vertices = triangle_vertices.view(num_triangles, 3, 3)  # shape(F, 3, 3)
    # vertex_n shape(F,3) positions in 3d for n-th vertex in triangle
    vertex_0, vertex_1, vertex_2 = torch.unbind(triangle_vertices, dim=1)
    normal_vector = torch.cross(vertex_1 - vertex_0, vertex_2 - vertex_0, dim=1)
    length = torch.sqrt(normal_vector.square().sum(dim=1, keepdim=True))  # shape (F,1)
    normal_vector /= length
    return normal_vector


def getBSPositions(A, restPos):
    numBS = A.shape[0] // 3
    numVertices = A.shape[1]

    aDeltas = A.reshape(numBS, 3, numVertices).permute(0, 2, 1).reshape(-1, 3)
    positionsBS = torch.tile(restPos, (numBS, 1))
    return positionsBS + aDeltas


def getBSTriangles(A, triangles):
    numBS = A.shape[0] // 3
    numVertices = A.shape[1]

    offsets = torch.arange(0, numVertices * numBS, numVertices, device="cpu")
    trianglesBS = torch.tile(triangles.unsqueeze(2), (1, 1, numBS)) + offsets
    return trianglesBS.permute(2, 0, 1).reshape(-1, 3)


# -----------------------------------edge angle-----------------------------------
def calcFaceCenters(v, f):
    num_triangles = f.shape[0]
    triangle_vertices = v[f.view(-1), :]  # build from value and indices  shape(F*3, 3)
    triangle_vertices = triangle_vertices.view(num_triangles, 3, 3)  # shape(F, 3(v), 3)

    face_center = triangle_vertices.sum(dim=1) * (1.0 / 3.0)
    return face_center


def calcEdgeAngles(v, f):
    _, _, ef = igl.edge_topology(toNumpy(v), toNumpy(f))
    ef = toTorchCpu(ef)

    normals = calcNormals(v, f)
    # remove border edges
    ef = ef[(ef[:, 0] != -1) & (ef[:, 1] != -1)]
    face0, face1 = torch.unbind(ef, dim=1)

    normal0 = normals[face0]
    normal1 = normals[face1]

    dot = torch.sum(normal0 * normal1, dim=1)
    dot[dot > 1.0] = 1.0
    dot[dot < -1.0] = -1.0
    angles = torch.acos(dot)

    centers = calcFaceCenters(v, f)
    center01 = centers[face1] - centers[face0]
    sign = -torch.sum(normal0 * center01, dim=1)
    angles[sign < 0.0] *= -1.0
    return angles


def calculateAverageEdgeAngularDiff(A, B, C, restPos, triangles):
    restPosTorch = toTorchCpu(restPos)
    ACpu = A.cpu()
    posA = getBSPositions(ACpu, restPosTorch)
    posBC = getBSPositions(B.cpu() @ C.cpu(), restPosTorch)
    trianglesBS = getBSTriangles(ACpu, toTorchCpu(triangles))

    edgeAnglesA = calcEdgeAngles(posA, trianglesBS)
    edgeAnglesBC = calcEdgeAngles(posBC, trianglesBS)
    edgeAngularDiff = edgeAnglesA - edgeAnglesBC

    return edgeAngularDiff.abs().mean()
