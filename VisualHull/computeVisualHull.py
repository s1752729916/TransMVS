import os
import numpy as np
import trimesh as trm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import measure
from VisualHull.loladCameras import loadCameraParams
from loadMasks import loadMasks
#-- 1. configure
jsonPath = '/media/smq/移动硬盘/Research/TransMVS/synthetic/cow/json'
maskPath = '/media/smq/移动硬盘/Research/TransMVS/synthetic/cow/masks'
checkFolder = '/media/smq/移动硬盘/Research/TransMVS/check'
resolution = 0.01
minX,maxX = -2,2
minY,maxY = 2,8
minZ,maxZ = -2,2
imgH = 1028
imgW = 1232

debug = True

#-- 2. get camera positions and masks
num,K,M,KM,origins,targets,ups = loadCameraParams(jsonPath)
masks = loadMasks(maskPath)
#-- 3. create voxels
y,x, z = np.meshgrid(
    np.linspace(minY, maxY, int((maxY-minY)/resolution)),
    np.linspace(minX, maxX, int((maxX-minX)/resolution)),
    np.linspace(minZ, maxZ, int((maxZ-minZ)/resolution))) # (256,256,256)
x = x[:, :, :, np.newaxis] # (256,256,256,1)
y = y[:, :, :, np.newaxis] # (256,256,256,1)
z = z[:, :, :, np.newaxis] # (256,256,256,1)
coord = np.concatenate([x, y, z], axis=3) # (256,256,256,3), 保存了所有点的三个坐标
volume = -np.ones(x.shape).squeeze() # (256,256,256), 保存体素占有的信息，值表示该体素占有信息

#-- 4. start to build voxels
for i in range(num):
    print('Processing {}/{} mask:'.format(i+1, num))
    seg = masks[:,:,i]
    mask = seg.reshape(imgH*imgW)

    f = K[:, :, i][0, 0]
    cx = K[:, :, i][0, 2]
    cy = K[:, :, i][1, 2]


    Rot = M[:,0:3,i]
    t = M[:,3,i].reshape([1,1,1,3,1])
    coordCam = np.matmul(Rot, np.expand_dims(coord, axis=4)) + t
    coordCam = coordCam.squeeze(4)
    xCam = coordCam[:, :, :, 0] / coordCam[:, :, :, 2]
    yCam = coordCam[:, :, :, 1] / coordCam[:, :, :, 2]


    xId = -xCam*f + cx
    yId = -yCam*f + cy # 这里代码里用的-yCam*f，不知道为什么，先试一下吧 becauuse the camera coordinate(y is direction toward to up) is different with pixel coordinate(y is toward to bottom of the image)
    xInd = np.logical_and(xId>=0,xId<imgW-0.5)
    yInd = np.logical_and(yId>=0,yId<imgH-0.5)
    imInd = np.logical_and(xInd,yInd) # x,y都在范围内的点(256,256,256),在范围内表示为1，不在表示为0

    xImId = np.round(xId[imInd]).astype(np.int32) #对x,y都在范围内的索引取整 返回n维向量，n维满足条件的点的个数，值为该点的x坐标(像素坐标系中)
    yImId = np.round(yId[imInd]).astype(np.int32)
    # print(np.max(xImId))
    # print(np.max(yImId))
    maskInd = mask[yImId*imgW + xImId] # 取出所有在该图片内的点的mask值，大小为n的向量，表示该点值为
    volumeInd = imInd.copy() #体素索引
    volumeInd[imInd==1] = maskInd   #记值，因为Ind包含了所有在图片范围内的点，不管是不是mask都会标记Ind=1
    a = np.sum(volumeInd==1)
    volume[volumeInd==0] = 1 # 表示占有，当volumeInd的值为0的时候表示占有,为什么mask值为0的时候是占有呢,确实occupied voxel指代的是除了模型之外的体素，可能是marching cube算法所要求的。

    print('Occupied voxel: %d' % np.sum((volume > 0).astype(np.float32)))
    verts, faces, normals, _ = measure.marching_cubes(volume)
    # convert coordinates from voxel to MinMax bounds
    num_x = int((maxX-minX)/resolution)
    num_y = int((maxY-minY)/resolution)
    num_z = int((maxZ-minZ)/resolution)
    verts[:,0] = (verts[:,0]/num_x)*(maxX-minX)+minX
    verts[:,1] = (verts[:,1]/num_y)*(maxY-minY)+minY
    verts[:,2] = (verts[:,2]/num_z)*(maxZ-minZ)+minZ
    print('Vertices Num: %d' % verts.shape[0])
    print('Normals Num: %d' % normals.shape[0])
    print('Faces Num: %d' % faces.shape[0])

    mesh = trm.Trimesh(vertices=verts, faces=faces)
    if(debug ==True):
        mesh.export(os.path.join(checkFolder, str(i).zfill(3)+'-check.ply'))
