import json
import glob
import os
import numpy as np
def loadCameraParams(jsonPath):
    # 从json文件中读取相机内参与外参
    # jsonPath: 存放json文件的根目录
    # return: N,K(3,3,N), M(3,4,N), KM(3,4,N)

    searchStr = os.path.join(jsonPath,'*.json')
    jsonLists = glob.glob(searchStr)
    num = len(jsonLists)
    print('Found %d json files' % num)
    K = np.zeros([3,3,num])
    M = np.zeros([3,4,num])
    KM = np.zeros([3,4,num])
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
    return num,K,M,KM
if __name__ =='__main__':
    jsonPath = 'F:\\Research\\TransMVS\\synthetic\\bear\\json'
    loadCameraParams(jsonPath)