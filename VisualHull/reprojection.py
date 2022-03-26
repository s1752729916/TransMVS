# 已知mesh,相机内参K,相机外参M,将mesh重投影到像素坐标系当中去
import trimesh as trm
import trimesh.scene
import psbody.mesh
from VisualHull.loladCameras import loadCameraParams
import numpy as np
from psbody.mesh.visibility import visibility_compute

def reprojection(mesh,origin,target):
    # return: mask,vertexNormal,faceNormal

    verts = mesh.vertices
    vertNormal = mesh.vertex_normals
    faceNormal = mesh.face_normals
    psbody.mesh.Mesh()



if __name__ =='__main__':
    jsonPath = '/media/smq/移动硬盘/Research/TransMVS/synthetic/bear/json'
    trimesh = trm.load('/media/smq/移动硬盘/Research/TransMVS/check/017-check.ply')
    b =  trimesh.vertices

    mesh = psbody.mesh.Mesh(filename='/media/smq/移动硬盘/Research/TransMVS/check/017-check.ply')
    num,K,M,KM,origin,target,up = loadCameraParams(jsonPath)
    a = mesh.v
    cam = origin[:,17]-target[:,17]
    new_mesh = mesh.visibile_mesh(camera=np.array([cam]))
    new_mesh.show()
    # vis = vis.squeeze()
    # print('vis.shape',vis)
    print('vers:',mesh.v.shape)
    # cam = target[:,0]-origin[:,0]
    print(cam)
    print(K[:,:,0])
    camera = trimesh.scene.Camera(name = 'camera', resolution=[1232,1028],focal = [1.85426858e+03,1.85426858e+03])
    print(camera.K)
