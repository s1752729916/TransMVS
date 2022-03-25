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
    jsonPath = 'F:\\Research\\TransMVS\\synthetic\\bear\\json'
    # mesh = trm.load('F:\\Research\\TransMVS\\check\\017-check.ply')
    mesh = psbody.mesh.Mesh(filename='F:\\Research\\TransMVS\\check\\017-check.ply')
    num,K,M,KM,origin,target,up = loadCameraParams(jsonPath)


    # cam = target[:,0]-origin[:,0]
    cam = np.array([0,0,0])
    print(cam)
    visible_mesh = mesh.visibile_mesh(cam)
    visible_mesh.show()
    print(K[:,:,0])
    camera = trimesh.scene.Camera(name = 'camera', resolution=[1232,1028],focal = [1.85426858e+03,1.85426858e+03])
    print(camera.K)
