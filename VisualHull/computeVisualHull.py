import os
import numpy as np
import trimesh as trm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import measure
from VisualHull.loladCameras import loadCameraParams
from loadMasks import loadMasks
#-- 1. configure
jsonPath = 'F:\\Research\\TransMVS\\synthetic\\bear\\json'
maskPath = 'F:\\Research\\TransMVS\\synthetic\\bear\\masks'
checkFolder = 'F:\\Research\\TransMVS\\check'
resolution = 0.01
minX,maxX = -2,2
minY,maxY = 3,6
minZ,maxZ = -2,2
imgH = 1028
imgW = 1232

debug = True

#-- 2. get camera positions and masks
num,K,M,KM,origin,target,up = loadCameraParams(jsonPath)
masks = loadMasks(maskPath)
#-- 3. create voxels
x, y, z = np.meshgrid(
    np.linspace(minX, maxX, int((maxX-minX)/resolution)),
    np.linspace(minY, maxY, int((maxY-minY)/resolution)),
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
    fx = K[:,:,i][0,0]
    fy = K[:,:,i][1,1]
    cx = K[:,:,i][0,2]
    cy = K[:,:,i][1,2]
    Rot = M[:,0:3,i]
    Trans = M[:,3,i]
    Trans = Trans.reshape([1,1,1,3,1]) # 转换成这个样子是为了可以后coord直接相加
    coordCam = np.matmul(Rot,np.expand_dims(coord,axis=4)) + Trans # 先将coord转换成(256,256,256,3,1)，再与Rot相乘，得到(256,256,256,3,1)，再与Trans相加，这步是将世界系的体素坐标转换到相机坐标系中去
    coordCam = coordCam.squeeze(4)
    xCam = coordCam[:,:,:,0]/coordCam[:,:,:,2] # 归一化过程
    yCam = coordCam[:,:,:,1]/coordCam[:,:,:,2]


    xId = xCam*fx + cx
    yId = yCam*fy + cy # 这里代码里用的-yCam*f，不知道为什么，先试一下吧
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

    verts, faces, normals, _ = measure.marching_cubes_lewiner(volume, 0)
    print('Vertices Num: %d' % verts.shape[0])
    print('Normals Num: %d' % normals.shape[0])
    print('Faces Num: %d' % faces.shape[0])

    # axisLen = float(resolution - 1) / 2.0
    # verts = (verts - axisLen) / axisLen * 1.7
    mesh = trm.Trimesh(vertices=verts, vertex_normals=normals, faces=faces)

    if(debug ==True):
        mesh.export(os.path.join(checkFolder, str(i).zfill(3)+'-check.ply'))

