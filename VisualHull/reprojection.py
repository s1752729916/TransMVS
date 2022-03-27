# 已知mesh,相机内参K,相机外参M,将mesh重投影到像素坐标系当中去
import trimesh as trm
import trimesh.scene
import psbody.mesh
from VisualHull.loladCameras import loadCameraParams
import numpy as np
from psbody.mesh.visibility import visibility_compute
import matplotlib.pyplot as plt


def reprojection(mesh,origin,target,up,K,H,W,M):
    """
    Reproject mesh to a given position camera view
    Parameters
    -----------
    mesh: psbody.mesh
    origin: np.array [3,]
        lookat origin of camera
    K: np.array [3,3]
        intrinsic of the camera
    M: np.array [3,4]
        extrinsic of the camera
    H: int
        height of tha image
    W: int
        width of the image

    Returns
    -----------
    file_obj : file-like object
      Contains data

    """
    # calculate the visible mesh
    visible_mesh = mesh.visibile_mesh(camera=np.array([origin]))
    # visible_mesh.show()

    # get params
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    yAxis = up / np.sqrt(np.sum(up * up))
    zAxis = target - origin
    zAxis = zAxis / np.sqrt(np.sum(zAxis * zAxis))
    xAxis = np.cross(zAxis, yAxis)
    xAxis = xAxis / np.sqrt(np.sum(xAxis * xAxis))
    Rot = np.stack([xAxis, yAxis, zAxis], axis=0)
    print('look at:',Rot)
    # reproject vertices to pixel coordinate
    verts = visible_mesh.v
    verts_cam_coord = np.matmul(Rot,np.expand_dims(verts-origin, axis=2))
    verts_cam_coord = verts_cam_coord.squeeze(2)
    xCam = verts_cam_coord[:,0]/verts_cam_coord[:,2]
    yCam = verts_cam_coord[:,1]/verts_cam_coord[:,2]
    xPixel =  np.round(xCam*fx + cx).astype(np.int32)
    yPixel = np.round(-yCam*fy + cy).astype(np.int32)
    img = np.zeros([H,W]).astype(np.uint8)
    img[yPixel,xPixel] = 255
    plt.figure()
    plt.imshow(img,cmap='gray')
    plt.show()

    # reproject vertex normals to cam coordinate
    Rot = M[:,0:3]
    print('M:',Rot)
    triMesh = trm.Trimesh(vertices=mesh.v, faces=mesh.f)
    vertNormals = triMesh.vertex_normals
    normalsCam = np.matmul(Rot,np.expand_dims(vertNormals, axis=2))
    normalsCam = normalsCam.squeeze(2)
    print(np.linalg.norm(normalsCam[5,:]))











if __name__ =='__main__':
    jsonPath = '/media/smq/移动硬盘/Research/TransMVS/synthetic/bear/json'
    trimesh = trm.load('/media/smq/移动硬盘/Research/TransMVS/check/017-check.ply')
    b =  trimesh.vertices

    fig = plt.figure()
    import imageio
    mask = imageio.imread('/media/smq/移动硬盘/Research/TransMVS/synthetic/bear/masks/005-view.png')
    plt.imshow(mask)
    mesh = psbody.mesh.Mesh(filename='/media/smq/移动硬盘/Research/TransMVS/check/017-check.ply')
    print(mesh.v.shape)
    num,K,M,KM,origin,target,up = loadCameraParams(jsonPath)
    reprojection(mesh = mesh,origin = origin[:,5],target = target[:,5],up = up[:,5],  K = K[:,:,5],H = 1028,W = 1232,M = M[:,:,5])



    # # calculate the visible mesh
    # visible_mesh = mesh.visibile_mesh(camera=np.array([origin]))
    # visible_mesh.show()
    # # reproject vertices to pixel coordinate
    #
    # fx = K[0, 0]
    # fy = K[1, 1]
    # cx = K[0, 2]
    # cy = K[1, 2]
    # print(cy)
    #
    # R = M[:,0:3]
    # t = M[:,3].reshape([1,3,1])
    # KM = np.matmul(K, M)
    # verts = visible_mesh.v
    # verts_cam_coord = np.matmul(R,np.expand_dims(verts, axis=2)) + t   # R*Pw + t
    # verts_cam_coord = verts_cam_coord.squeeze(2)
    #
    # xCam = verts_cam_coord[:,0]/verts_cam_coord[:,2]
    # yCam = verts_cam_coord[:,1]/verts_cam_coord[:,2]
    #
    #
    # xPixel =  np.round(xCam*fx + cx).astype(np.int32)
    # yPixel = np.round(-yCam*fy + cy).astype(np.int32)
    #
    # img = np.zeros([H,W]).astype(np.uint8)
    # img[yPixel,xPixel] = 255
    # plt.imshow(img,cmap='gray')
    # plt.show()
