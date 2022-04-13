# -*-coding:utf-8-*-
import imageio
import numpy as np
import os
import glob


def loadNormals(normalsPath):
    # return: imgs: [H,W,N] np.uint8
    searchStr = os.path.join(normalsPath, '*.png')
    pngLists = glob.glob(searchStr)
    num = len(pngLists)
    print('Found %d png files' % num)
    img = imageio.imread(pngLists[0])
    imgs = np.zeros([img.shape[0], img.shape[1],3, num]).astype(np.uint8)
    for i in range(0, num):
        imgs[:, :, :, i] = imageio.imread(pngLists[i])
        # import matplotlib.pyplot as plt
        # plt.imshow(imgs[:,:,:,i])
        # plt.show()
    return imgs


if __name__ == '__main__':
    aolpPath = '/media/smq/移动硬盘/Research/TransMVS/synthetic/bear/normals-png'
    loadNormals(aolpPath)
# -*-coding:utf-8-*-
