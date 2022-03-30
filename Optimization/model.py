# -*-coding:utf-8-*-
import torch
import trimesh as trm
from VisualHull.loladCameras import loadCameraParams
from VisualHull.reprojection import reprojection
import numpy as np
import psbody.mesh


class Model():

    def __init__(self, mesh, K, M, origin, verbose=False):
        """
        Return a torch model for processing optimazation problem
        Parameters
        -----------
        mesh: psbody.mesh
        K: np.array [3,3,N]
        """
        super(Model, self).__init__()
        self.rawMesh = mesh
        self.vertices = torch.from_numpy(mesh.vertices)  # [n,3]
        self.vertices.requires_grad = True
        self.faces = torch.from_numpy(mesh.faces)  # [m,3]
        self.faces.requires_grad = False
        self.faces_by_vertex = mesh.vertex_faces  # [m,max number of faces for a single vertex]

        self.n = self.vertices.shape[0]  # num of total verteices
        self.m = self.faces.shape[0]  # num of total faces
        self.N = K.shape[2]  # num of views
        self.N = 2
        self.origin = origin  # camera origin point
        self.Ks = K  # numpy
        self.Ms = M  # numpy

        self.H = 1028
        self.W = 1232
        self.verbose = verbose

    def computeVisibility(self):
        # after every backward shouble recompute visibility for all views
        mesh_temp = trm.Trimesh(vertices=self.vertices.detach().numpy(), faces=self.faces.detach().numpy())
        self.visibilityVerts = []  # [n,1,N]
        self.vert_xPixels = []  # [n,N]
        self.vert_yPixels = []  # [n,N]
        self.face_xPixels = []  # [n,N]
        self.face_yPixels = []  # [n,N]
        self.visibilityFaces = []  # [m,N], contains visibility of faces
        self.visibleFacesNewIndices = []  # visible faces with new indices(rearrange visible verts)
        self.visibleMesh = []  # visible triMesh objects
        for i in range(0, self.N):
            if (self.verbose):
                print('computeVisibility %d' % i)
            n, vert_xPixel, vert_yPixel, face_xPixel, face_yPixel, normalsCam, visible_mesh, visibilityVerts, visibilityFaces = reprojection(
                mesh=mesh_temp,
                origin=origin[:,
                       i],
                K=self.Ks[:, :,
                  i],
                M=self.Ms[:, :,
                  i])
            self.vert_xPixels.append(vert_xPixel)
            self.vert_yPixels.append(vert_yPixel)
            self.face_xPixels.append(face_xPixel)
            self.face_yPixels.append(face_yPixel)
            self.visibilityVerts.append(visibilityVerts)
            self.visibilityFaces.append(visibilityFaces)
            self.visibleFacesNewIndices.append(visible_mesh.faces)
            self.visibleMesh.append(visible_mesh)
            # print(self.faces[np.where(self.visibilityFaces[i])])
            # self.showMesh(self.vertices[np.where(self.visibilityVerts[i])].detach().numpy(),f =self.visibleFacesNewIndices[i])

    def calculateNormals(self):
        # calculate face and vertex normals

        self.visibleFaceNormals = []
        self.visibleVertNormals = []
        # -- 1. calculate face normals
        face_normals = torch.cross(self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]],
                                   self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]],
                                   dim=1)  # all face normals (m,3)
        face_normals = face_normals / torch.linalg.norm(face_normals, ord=2, dim=1).reshape(-1, 1)

        for i in range(0, self.N):
            if (self.verbose):
                print('calculateNormals %d' % i)
            visible_vertex_index = np.where(self.visibilityVerts[i])
            visible_vertex_normals = torch.zeros([self.visibilityVerts[i].sum(), 3])

            # calculate visible vertex normals
            # ========================================
            # efficient of vertex normals calculation
            vertex = visible_vertex_index[0]  # (n_visible,1) contain indices of visible vertices
            faces = torch.from_numpy(self.faces_by_vertex[vertex, :]).type(torch.long)
            a = face_normals[faces, :]
            ax = a[:, :, 0]
            ay = a[:, :, 1]
            az = a[:, :, 2]
            ax[torch.where(faces < 0)] = torch.nan
            ay[torch.where(faces < 0)] = torch.nan
            az[torch.where(faces < 0)] = torch.nan
            ax = torch.nanmean(ax, dim=1)
            ay = torch.nanmean(ay, dim=1)
            az = torch.nanmean(az, dim=1)
            visible_vertex_normals[:, 0] = ax
            visible_vertex_normals[:, 1] = ay
            visible_vertex_normals[:, 2] = az
            # ========================================

            # for k in range(0, self.visibilityVerts[i].sum()):
            #     # process every visible vertex
            #     vertex = visible_vertex_index[0][k]  # vertex value
            #     faces = self.faces_by_vertex[vertex, :]  # faces with this vertex
            #     visible_vertex_normals[k, :] = torch.mean(face_normals[faces[np.where(faces >= 0)], :], dim=0)W
            visible_vertex_normals = visible_vertex_normals / torch.linalg.norm(visible_vertex_normals, ord=2,
                                                                                dim=1).reshape(-1, 1)
            self.visibleVertNormals.append(visible_vertex_normals)
            self.visibleFaceNormals.append(face_normals[np.where(self.visibilityFaces[i])].float())

    def reprojectNormals(self):
        # reproject vertex normals and face normals to image normal coordinate, the face normals and vertex normals should calculated by tensor operation
        self.projectedVertNormals = []
        self.projectedFaceNormals = []
        for i in range(0, self.N):
            if (self.verbose):
                print('reprojectNormals %d' % i)

            M = self.Ms[:, :, i]
            Rot = torch.from_numpy(M[:, 0:3]).float()
            vertNormalsCam = torch.matmul(Rot, self.visibleVertNormals[i].view(-1, 3, 1))
            vertNormalsCam = torch.squeeze(vertNormalsCam, dim=2)
            vertNormalsCam[:, 0] = -vertNormalsCam[:, 0]
            vertNormalsCam[:, 2] = -vertNormalsCam[:, 2]

            faceNormalsCam = torch.matmul(Rot, self.visibleFaceNormals[i].view(-1, 3, 1))
            faceNormalsCam = torch.squeeze(faceNormalsCam, dim=2)
            faceNormalsCam[:, 0] = -faceNormalsCam[:, 0]
            faceNormalsCam[:, 2] = -faceNormalsCam[:, 2]

            self.projectedVertNormals.append(vertNormalsCam)
            self.projectedFaceNormals.append(faceNormalsCam)
        self.showNormal(self.projectedVertNormals[0].detach().numpy(), xPixel=self.vert_xPixels[0],
                        yPixel=self.vert_yPixels[0])

    def computeGeometricSmoothLoss(self):
        # compute the geometric smooth loss term
        pass

    def showMesh(self, v, f):
        # self.showMesh(self.vertices[np.where(self.visibilityVerts[i])].detach().numpy(),f=self.visibleFacesNewIndices[i])

        mesh_temp = psbody.mesh.Mesh(v=v, f=f)
        mesh_temp.show()

    def showNormal(self, projectedNormal, xPixel, yPixel):
        import matplotlib.pyplot as plt
        normals = np.zeros([self.H, self.W, 3])
        normals[yPixel, xPixel] = projectedNormal
        normals = (normals + 1) * 127.5
        plt.imshow(normals.astype(np.uint8))
        plt.show()


if __name__ == "__main__":
    jsonPath = '/media/smq/移动硬盘/Research/TransMVS/synthetic/bear/json'
    trimesh = trm.load('/media/smq/移动硬盘/Research/TransMVS/check/017-check.ply')
    import imageio

    mask = imageio.imread('/media/smq/移动硬盘/Research/TransMVS/synthetic/bear/masks/005-view.png')
    num, K, M, KM, origin, target, up = loadCameraParams(jsonPath)
    model = Model(mesh=trimesh, K=K, M=M, origin=origin, verbose=False)

    model.computeVisibility()
    model.calculateNormals()
    model.reprojectNormals()
