# -*-coding:utf-8-*-
import imageio
import numpy as np
import os
import glob
def loadAoLP(aolpPath):
    # return: imgs: [H,W,N] np.uint8
    searchStr = os.path.join(aolpPath,'*.png')
    pngLists = glob.glob(searchStr)
    num = len(pngLists)
    print('Found %d png files' % num)
    img = imageio.imread(pngLists[0])
    imgs = np.zeros([img.shape[0],img.shape[1],num]).astype(np.uint8)
    for i in range(0,num):
        imgs[:,:,i] = imageio.imread(pngLists[i])
        import matplotlib.pyplot as plt
        # plt.imshow(imgs[:,:,i],cmap='gray')
        # plt.show()
    return imgs
if __name__ =='__main__':
    aolpPath = '/media/smq/移动硬盘/Research/TransMVS/synthetic/bear/params/AoLP'
    loadAoLP(aolpPath)