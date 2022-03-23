import json
import glob
import os
import numpy as np
import imageio
def loadMasks(maskPath):
    # 读取mask文件
    # maskPath: 存放mask文件的根目录
    # return: masks[H,W,N]

    searchStr = os.path.join(maskPath,'*.png')
    maskLists = glob.glob(searchStr)
    num = len(maskLists)
    print('Found %d mask files' % num)
    H = 1028
    W = 1232
    masks = np.zeros([H,W,num])
    for i in range(0,num):
        img = imageio.imread(maskLists[i])
        masks[:,:,i] = img


    return masks
if __name__ =='__main__':
    jsonPath = 'F:\\Research\\TransMVS\\synthetic\\bear\\masks'
    masks = loadMasks(jsonPath)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(masks[:,:,12])
    plt.show()