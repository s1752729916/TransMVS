import json
import glob
import os
import numpy as np
def loadCameraParams(jsonPath):
    # 从json文件中读取相机内参与外参
    # jsonPath: 存放json文件的根目录
    # return: N,K(3,3,N), M(3,4,N), KM(3,4,N), lookat_origin(3,N), lookat_target(3,N), lookat_up(3,N)

    searchStr = os.path.join(jsonPath,'*.json')
    jsonLists = glob.glob(searchStr)
    num = len(jsonLists)
    print('Found %d json files' % num)
    K = np.zeros([3,3,num])
    M = np.zeros([3,4,num])
    KM = np.zeros([3,4,num])
    origin = np.zeros([3,num])
    target = np.zeros([3,num])
    up = np.zeros([3,num])
    for i in range(0,num):
        with open(jsonLists[i], 'r') as f:
            data = json.load(f)

        # 计算内参
        resx = int(data[0]['intrinsic'][0]['resx'])
        resy = int(data[0]['intrinsic'][0]['resy'])
        focal = 0.05
        diag = (0.036**2 + 0.024**2)**0.5
        h_relative = resy/resx
        sensor_width = np.sqrt(diag**2 / (1 + h_relative**2))
        dx = sensor_width/resx
        fx = focal/dx
        fy = fx
        cx = resx/2
        cy = resy/2
        K[:,:,i] = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

        # 读取外参
        extrinsic = data[0]['extrinsic']
        extrinsic = np.array(extrinsic)
        extrinsic = np.linalg.inv(extrinsic)
        M[:,:,i] = np.array(extrinsic)[0:3,:]
        KM[:,:,i] = np.matmul(K[:,:,i],M[:,:,i])
        # 读取look at
        look_origin = data[0]['extrinsic_lookat'][0]['origin'].split(',')
        look_origin = np.array(look_origin).astype(np.float)
        look_target = data[0]['extrinsic_lookat'][0]['target'].split(',')
        look_target = np.array(look_target).astype(np.float)
        look_up = data[0]['extrinsic_lookat'][0]['up'].split(',')
        look_up = np.array(look_up).astype(np.float64)
        origin[:,i] = look_origin
        target[:,i] = look_target
        up[:,i] = look_up


    return num,K,M,KM,origin,target,up
if __name__ =='__main__':
    jsonPath = '/media/smq/移动硬盘/Research/TransMVS/synthetic/cow/json'
    loadCameraParams(jsonPath)