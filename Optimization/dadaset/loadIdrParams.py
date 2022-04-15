# -*-coding:utf-8-*-
import numpy as np
import cv2
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)


    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]
    return intrinsics, pose
def loadIdrParams(cam_file,n):
    intrinsics_all = []
    pose_all = []
    K = np.zeros([3,3,n])
    M = np.zeros([3,4,n])
    KM = np.zeros([3,4,n])
    origin = np.zeros([3,n])

    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n)]
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)

        intrinsics_all.append(intrinsics)
        pose_all.append(np.linalg.inv(pose))
    for i in range(0,n):
        K[:,:,i] = intrinsics_all[i][0:3,0:3]
        M[:,:,i] = pose_all[i][0:3,:]
        KM[:, :, i] = np.matmul(K[:, :, i], M[:, :, i])
        origin[:,i] = pose_all[i][0:3,3]
        print('K:',K[:,:,i])
        print('M:',M[:,:,i])

    return K,M,KM,origin


if __name__ == '__main__':
    cam_file = '/media/smq/移动硬盘/Research/TransMVS/synthetic/cow/cameras_new.npz'
    loadIdrParams(cam_file,40)
