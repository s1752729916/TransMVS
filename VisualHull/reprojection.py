# 已知mesh,相机内参K,相机外参M,将mesh重投影到像素坐标系当中去
import trimesh as trm
import trimesh.scene
import psbody.mesh
from VisualHull.loladCameras import loadCameraParams
import numpy as np
import matplotlib.pyplot as plt
import time





def reprojection(mesh,origin,K,M):
    """
    Reproject mesh to a given position camera view
    Parameters
    -----------
    mesh: psbody.mesh
    origin : np.array [3,]
        lookat origin of camera
    K : np.array [3,3]
        intrinsic of the camera
    M : np.array [3,4]
        extrinsic of the camera
    H : int
        height of tha image
    W : int
        width of the image
    Returns
    -----------
    n : int
      number of visible vertices, vertex normals
    vert_xPixel : np.array [n,]
        x of visible vertices in pixel coordinate(u)
    vert_yPixel : np.array [n,]
        y of visible vertices in pixel coordinate(v)
    face_xPixel : np.array [n,]
        x of visible faces in pixel coordinate(u)
    face_yPixel : np.array [n,]
        y of visible faces in pixel coordinate(v)
    normalsCam : np.array [n,3]
        visible vertex normals reproject to certain image coordinate(x --> right, y --> up,z --> outside)
    visibleMesh : trimesh.Mesh
        visible mesh
    """

    # calculate the visible mesh
    # start =  time.time()
    mesh_temp = psbody.mesh.Mesh(v = mesh.vertices,f = mesh.faces)

    visible_mesh = mesh_temp.visibile_mesh(camera=np.array([origin]))
    # print('end:',time.time()-start)

    visibilityVerts = mesh_temp.vertex_visibility(camera=np.array([origin])) # visibility matrix for original input mesh
    n_visibility = visibilityVerts.sum() # num of visible vertices

    # visible faces in old indices
    faces_to_keep = np.where(visibilityVerts[mesh.faces[:, 0]] * visibilityVerts[mesh.faces[:, 1]] * visibilityVerts[mesh.faces[:, 2]])
    visibilityFaces = np.zeros([mesh.faces.shape[0]]).astype(np.int32)
    visibilityFaces[faces_to_keep] = 1

    # visible_mesh.show()
    triMesh = trm.Trimesh(vertices=visible_mesh.v, faces=visible_mesh.f,process=False)
    # get params
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    Rot = M[:,0:3]

    t = M[:,3].reshape([1,3,1])
    # reproject vertices to pixel coordinate
    verts = triMesh.vertices
    verts_cam_coord = np.matmul(Rot,np.expand_dims(verts, axis=2)) + t
    verts_cam_coord = verts_cam_coord.squeeze(2)
    xCam = verts_cam_coord[:,0]/verts_cam_coord[:,2]
    yCam = verts_cam_coord[:,1]/verts_cam_coord[:,2]
    xPixel =  np.round(-xCam*fx + cx).astype(np.int32)
    yPixel = np.round(-yCam*fy + cy).astype(np.int32)

    # reproject position of faces to pixel coordinate
    face_positions = np.mean(verts[triMesh.faces],axis = 1)
    faces_cam_coord = np.matmul(Rot,np.expand_dims(face_positions, axis=2)) + t
    faces_cam_coord = faces_cam_coord.squeeze(2)
    faces_xCam = faces_cam_coord[:,0]/faces_cam_coord[:,2]
    faces_yCam = faces_cam_coord[:,1]/faces_cam_coord[:,2]
    faces_xPixel =  np.round(-faces_xCam*fx + cx).astype(np.int32)
    faces_yPixel = np.round(-faces_yCam*fy + cy).astype(np.int32)

    img = np.zeros([1028,1232]).astype(np.uint8)
    img[faces_yPixel,faces_xPixel] = 255
    # plt.figure()
    # plt.imshow(img,cmap='gray')
    # plt.show()

    # reproject vertex normals to cam coordinate
    visible_mesh.estimate_vertex_normals()
    vertNormals = triMesh.vertex_normals
    normalsCam = np.matmul(Rot,np.expand_dims(vertNormals, axis=2))
    normalsCam = normalsCam.squeeze(2)
    normalsCam[:,0] = -normalsCam[:,0] # conver to normal space
    normalsCam[:,2] = -normalsCam[:,2]
    # normal_img = np.zeros([H,W,3]).astype(np.uint8)
    # normal_img[yPixel,xPixel,0] = ((normalsCam[:,0] + 1)*127.5).astype(np.uint8) # for display
    # normal_img[yPixel,xPixel,1] = ((normalsCam[:,1] + 1)*127.5).astype(np.uint8)
    # normal_img[yPixel,xPixel,2] = ((normalsCam[:,2] + 1)*127.5).astype(np.uint8)

    if(xPixel.shape[0]==vertNormals.shape[0]==n_visibility):
        n = n_visibility
    else:
        raise RuntimeError('reprojection: error of visible nums')


    return n,xPixel,yPixel,faces_xPixel,faces_yPixel,normalsCam,triMesh,visibilityVerts,visibilityFaces









if __name__ =='__main__':
    jsonPath = '/media/smq/移动硬盘/Research/TransMVS/synthetic/bear/json'
    trimesh = trm.load('/media/smq/移动硬盘/Research/TransMVS/check/017-check.ply')


    fig = plt.figure()
    import imageio
    mask = imageio.imread('/media/smq/移动硬盘/Research/TransMVS/synthetic/bear/masks/005-view.png')
    plt.imshow(mask)
    mesh = psbody.mesh.Mesh(filename='/media/smq/移动硬盘/Research/TransMVS/check/017-check.ply')

    num,K,M,KM,origin,target,up = loadCameraParams(jsonPath)
    n,xPixel,yPixel,faces_xPixel,faces_yPixel,normalsCam,triMesh,visibilityVerts,visibilityFaces = reprojection(mesh = trimesh,origin = origin[:,0],  K = K[:,:,0],M = M[:,:,0])
    fig = plt.figure()
    normals = np.zeros([1028,1232,3])
    normals[yPixel,xPixel] = normalsCam
    normals= (normals+1)*127.5
    plt.imshow(normals.astype(np.uint8))
    plt.show()