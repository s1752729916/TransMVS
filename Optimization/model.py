# -*-coding:utf-8-*-
import os.path
import matplotlib.pyplot as plt
import torch
import trimesh as trm
from VisualHull.loladCameras import loadCameraParams
from VisualHull.reprojection import reprojection
import numpy as np
import psbody.mesh
from Optimization.loadAoLP import loadAoLP
import math

class Model():

    def __init__(self, mesh, aolps, K, M, origin, verbose=False):
        """
        Return a torch model for processing optimazation problem
        Parameters
        -----------
        mesh: psbody.mesh
        K: np.array [3,3,N]
        """
        super(Model, self).__init__()
        self.rawMesh = mesh
        self.newMesh = self.subdivision(self.rawMesh,iter=0)
        self.vertices = torch.from_numpy(self.newMesh.vertices)  # [n,3]
        self.vertices.requires_grad = True
        self.faces = torch.from_numpy(self.newMesh.faces)  # [m,3]
        self.faces.requires_grad = False

        self.n = self.vertices.shape[0]  # num of total verteices
        self.m = self.faces.shape[0]  # num of total faces
        self.N = K.shape[2]  # num of views

        self.origin = origin  # camera origin point
        self.Ks = K  # numpy
        self.Ms = M  # numpy
        self.aolps = aolps

        self.H = 1028
        self.W = 1232
        self.verbose = verbose

        # optimizer params
        self.lr = 5e-4

        # loss params
        self.q = 2.2
        self.k = 0.5
        self.tau_1 = 1
        self.tau_2 = 1


    def subdivision(self,rawMesh,iter = 4):
        newMesh = rawMesh
        for i in range(0,iter):
            newMesh = newMesh.subdivide()
        return newMesh
    def computeVisibility(self):
        # after every backward shouble recompute visibility for all views
        mesh_temp = trm.Trimesh(vertices=self.vertices.detach().numpy(), faces=self.faces.detach().numpy(),
                                process=False)
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
        faces_by_vertex = self.newMesh.vertex_faces
        self.visibleFaceNormals = []
        self.visibleVertNormals = []
        # -- 1. calculate face normals
        self.face_normals = torch.cross(self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]],
                                        self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]],
                                        dim=1)  # all face normals (m,3)
        self.face_normals = self.face_normals / torch.linalg.norm(self.face_normals, ord=2, dim=1).reshape(-1, 1)

        for i in range(0, self.N):
            if (self.verbose):
                print('calculateNormals %d' % i)
            visible_vertex_index = np.where(self.visibilityVerts[i])
            visible_vertex_normals = torch.zeros([self.visibilityVerts[i].sum(), 3])

            # calculate visible vertex normals
            # ========================================
            # efficient of vertex normals calculation
            vertex = visible_vertex_index[0]  # (n_visible,1) contain indices of visible vertices
            faces = torch.from_numpy(faces_by_vertex[vertex, :]).type(torch.long)
            a = self.face_normals[faces, :]
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
            #     faces = faces_by_vertex[vertex, :]  # faces with this vertex
            #     visible_vertex_normals[k, :] = torch.mean(face_normals[faces[np.where(faces >= 0)], :], dim=0)W
            visible_vertex_normals = visible_vertex_normals / torch.linalg.norm(visible_vertex_normals, ord=2,
                                                                                dim=1).reshape(-1, 1)
            self.visibleVertNormals.append(visible_vertex_normals)
            self.visibleFaceNormals.append(self.face_normals[np.where(self.visibilityFaces[i])].float())

    def reprojectNormals(self):
        # reproject vertex normals and face normals to image normal coordinate, the face normals and vertex normals should calculated by tensor operation
        self.projectedVertNormals = []
        self.projectedFaceNormals = []
        for i in range(0, self.N):
            if (self.verbose):
                print('reprojectNormals %d' % i)

            M = self.Ms[:, :, i]

            vertNormalsCam = self.projectNormals(normal_tensor=self.visibleVertNormals[i], M=M)

            faceNormalsCam = self.projectNormals(normal_tensor=self.visibleFaceNormals[i], M=M)

            self.projectedVertNormals.append(vertNormalsCam)
            self.projectedFaceNormals.append(faceNormalsCam)
            # self.showNormal(self.projectedVertNormals[i].detach().numpy(), xPixel=self.vert_xPixels[i],
            #                 yPixel=self.vert_yPixels[i])

    def projectNormals(self, normal_tensor, M):
        # project normals in world coor to image normal coordinate
        Rot = torch.from_numpy(M[:, 0:3]).float()
        normalCam = torch.matmul(Rot, normal_tensor.view(-1, 3, 1))
        normalCam = torch.squeeze(normalCam, dim=2)
        normalCam[:, 0] = -normalCam[:, 0]
        normalCam[:, 2] = -normalCam[:, 2]
        return normalCam

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(params=[self.vertices], lr=self.lr)

    def optimize(self):
        self.optimizer.zero_grad()
        loss = self.computeLoss()

        loss.backward()
        self.optimizer.step()


    def computeLoss(self):
        loss_pol = self.computePolarmetricLoss()*self.tau_1
        loss_gsm = self.computeGeometricSmoothLoss()*self.tau_2
        loss_all = loss_pol + loss_gsm
        print('loss_gsm:',loss_gsm)
        print('loss_pol:',loss_pol)
        print('loss_all:',loss_all)
        return loss_all

    def computePolarmetricLoss(self):
        # compute the polarmetric loss defined in Polarimetric Multi-View Inverse Rendering(2020,ECCV)
        loss_all = torch.zeros(self.faces.shape[0], 1)
        visibility = torch.zeros_like(torch.from_numpy(self.visibilityFaces[0].astype(np.float64)))
        for i in range(0, self.N):
            loss = torch.zeros(self.faces.shape[0],1)
            # -- 1. get azimith angles of visible veretex normals N(X)
            verts_normals_cam = self.projectedFaceNormals[i]
            # self.showNormal(verts_normals_cam.detach().numpy(),self.vert_xPixels[i],self.vert_yPixels[i])
            verts_azimuth_angles = torch.atan2(verts_normals_cam[:, 1], verts_normals_cam[:, 0])  # (-pi,pi)
            verts_azimuth_angles = torch.remainder(verts_azimuth_angles, torch.pi * 2)  # (0,2*pi)

            # a = verts_azimuth_angles.detach().numpy()
            # azimuth_img = verts_azimuth_angles.detach().numpy()
            # azimuth_img=np.expand_dims(azimuth_img, axis=1).repeat(3,axis=1)
            # azimuth_img = azimuth_img/np.pi
            # self.showNormal(azimuth_img,self.vert_xPixels[i],self.vert_yPixels[i])

            # -- 2. compute loss
            aolp = torch.from_numpy(self.aolps[:, :, i])
            aolp_corresponding = aolp[self.face_xPixels[i], self.face_yPixels[i]].type(
                torch.float64)  # get corresponding aolp for visible vertices
            aolp_corresponding = aolp_corresponding / 255 * torch.pi  # convert to (0,pi)

            aolp_1 = aolp_corresponding
            aolp_2 = aolp_corresponding + torch.pi
            aolp_3 = aolp_corresponding + torch.pi / 2
            aolp_4 = aolp_corresponding - torch.pi / 2
            aolp_1 = torch.remainder(aolp_1,torch.pi*2)  # convert all aolps to (0,2*pi)
            aolp_2 = torch.remainder(aolp_2,torch.pi*2)
            aolp_3 = torch.remainder(aolp_3,torch.pi*2)
            aolp_4 = torch.remainder(aolp_4,torch.pi*2)

            eta = torch.zeros(verts_azimuth_angles.shape[0], 4)
            eta[:, 0] = torch.min(torch.abs(verts_azimuth_angles - aolp_1),torch.pi*2-torch.abs(verts_azimuth_angles - aolp_1))
            eta[:, 1] = torch.min(torch.abs(verts_azimuth_angles - aolp_2),torch.pi*2-torch.abs(verts_azimuth_angles - aolp_2))
            eta[:, 2] = torch.min(torch.abs(verts_azimuth_angles - aolp_3),torch.pi*2-torch.abs(verts_azimuth_angles - aolp_3))
            eta[:, 3] = torch.min(torch.abs(verts_azimuth_angles - aolp_4),torch.pi*2-torch.abs(verts_azimuth_angles - aolp_4))

            eta, indices = torch.min(eta, dim=1)
            eta = 4 * eta / torch.pi
            theta = 1 - eta
            visibleLoss = (torch.exp(-self.k*theta)-math.exp(-self.k)).reshape(-1,1)

            loss[torch.where(torch.from_numpy(self.visibilityFaces[i].astype(np.int32)).type(torch.long))] = visibleLoss

            loss_all += loss
            visibility += self.visibilityFaces[i]

        loss_all = loss_all/(visibility).reshape(-1,1)
        loss_all[torch.where(visibility<1)] = 0
        loss_all = torch.sum(loss_all)
        return loss_all

    def computeGeometricSmoothLoss(self):
        # compute the geometric smooth loss term
        # -- 1. compute neighbors_mean_normals
        neighbors = torch.from_numpy(
            self.findNeighborFaces(self.vertices.detach().numpy(), self.faces.detach().numpy())).type(
            torch.long)  # (m,3*11)
        neighbors_mean_normals = torch.zeros([self.m, 3])

        ax = self.face_normals[neighbors, 0]
        ay = self.face_normals[neighbors, 1]
        az = self.face_normals[neighbors, 2]

        ax[torch.where(neighbors < 0)] = torch.nan
        ay[torch.where(neighbors < 0)] = torch.nan
        az[torch.where(neighbors < 0)] = torch.nan

        ax = torch.nanmean(ax, dim=1)
        ay = torch.nanmean(ay, dim=1)
        az = torch.nanmean(az, dim=1)

        neighbors_mean_normals[:, 0] = ax
        neighbors_mean_normals[:, 1] = ay
        neighbors_mean_normals[:, 2] = az
        neighbors_mean_normals = neighbors_mean_normals / torch.linalg.norm(neighbors_mean_normals, ord=2,
                                                                            dim=1).reshape(-1, 1)

        # -- 2. compute loss
        loss = self.face_normals * neighbors_mean_normals
        loss = torch.sum(loss, dim=1)
        eps = 1e-10
        loss = torch.clamp(loss, (-1.0 + eps), (1.0 - eps))
        loss = torch.pow(torch.acos(loss)/torch.pi,self.q)
        loss = torch.sum(loss)
        # visibleNormal = neighbors_mean_normals[torch.where(torch.from_numpy(self.visibilityFaces[0]).type(torch.long))]
        # visibleNormalCam = self.projectNormals(visibleNormal,self.Ms[:,:,0])
        # self.showNormal(visibleNormalCam.detach().numpy(),xPixel=self.face_xPixels[0],yPixel=self.face_yPixels[0])

        return loss

    def findNeighborFaces(self, v, f):
        # get neighbor faces indices of all faces
        mesh_temp = trm.Trimesh(vertices=v, faces=f, process=False)

        faces_by_vertex = mesh_temp.vertex_faces
        neighbors = faces_by_vertex[f].reshape(-1, 3 * faces_by_vertex.shape[1])
        return neighbors

    def findNeightborVertices(self, v, vertexIdx):
        # get neighbor vertices indices of all vertices
        pass

    def showMesh(self, v, f):
        # self.showMesh(self.vertices[np.where(self.visibilityVerts[i])].detach().numpy(),f=self.visibleFacesNewIndices[i])
        # mesh_temp = trm.Trimesh(v=v,f=f)
        mesh_temp = psbody.mesh.Mesh(v=v, f=f)
        mesh_temp.show()

    def saveMesh(self, v, f, path):
        mesh_temp = trm.Trimesh(vertices=v, faces=f)
        mesh_temp.export(path)

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
    checkPath = '/media/smq/移动硬盘/Research/TransMVS/optimize_check'
    aolpPath = '/media/smq/移动硬盘/Research/TransMVS/synthetic/bear/standard-params/AoLP'
    import imageio

    new_mesh = trm.Trimesh(vertices=trimesh.vertices)
    aolps = loadAoLP(aolpPath=aolpPath)
    mask = imageio.imread('/media/smq/移动硬盘/Research/TransMVS/synthetic/bear/masks/005-view.png')
    num, K, M, KM, origin, target, up = loadCameraParams(jsonPath)
    model = Model(mesh=trimesh, aolps=aolps, K=K, M=M, origin=origin, verbose=False)
    # model.showMesh(v=model.vertices.detach().numpy(),f=model.faces.detach().numpy())

    # model.showMesh(v=new_mesh.vertices,f=new_mesh.faces)

    model.saveMesh(v=model.vertices.detach().numpy(), f=model.faces.detach().numpy(),
                   path=os.path.join(checkPath, 'raw-check.ply'))
    for i in range(0, 100):
        print('#'*30)
        print('processing %d' % i)
        model.computeVisibility()
        model.calculateNormals()
        model.reprojectNormals()
        model.setup_optimizer()
        model.optimize()
        model.saveMesh(v=model.vertices.detach().numpy(), f=model.faces.detach().numpy(),
                       path=os.path.join(checkPath, str(i).zfill(3) + '-check.ply'))
        print('\n\n')


        # model.showMesh(v=model.vertices.detach().numpy(),f=model.faces.detach().numpy())
        # model.computeGeometricSmoothLoss()
