#============================================================#
############# SHUJI HEAD DETECTION Version 1.0 ###############
#============================================================#
#==========================2018-04-17========================#


import numpy as np
import cv2
import math
import time

def circular_block(origin, radius):
    block_size =4       # block尺寸大小
    # 3种半径的圆周ROI个数分别为8,12,16
    for r, N in zip((radius),(8,12,16)):
        x = [r * math.cos(2 * math.pi * (n+1) / N) for n in range(N)]
        y = [r * math.sin(2 * math.pi * (n+1) / N) for n in range(N)]

        for _x,_y in zip(x,y):
            # 小ROI矩形框的四个坐标
            rect = origin[0]+_x-block_size,origin[1]+_y-block_size,origin[0]+_x+block_size, origin[1]+_y+block_size
            yield list(map(lambda x:int(x),rect))


winSize = (100, 100)
blockSize = (20, 20)
blockStride = (10, 10)
cellSize = (10, 10)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
## fea added hog descriptor
def fea_hog(head_img, block_size):

    ## hog feature calculation
    descriptor = hog.compute(head_img)
    if descriptor is None:
        descriptor = []
    else:
        descriptor = descriptor.ravel()
    descriptor = descriptor.reshape(1, -1)
    # print("des=", descriptor[0])

    ## circle feature calculation
    rect = 50 - block_size, 50 - block_size, 50 + block_size, 50 + block_size
    # 中心小ROI矩形框图像 reference block
    m = np.mean(head_img[rect[1]:rect[3], rect[0]:rect[2]])  # 当前block的均值
    roi_img = head_img[rect[1]:rect[3], rect[0]:rect[2]].copy()
    rect = circular_block((50, 50), (3 * block_size - 4, 6 * block_size - 8, 9 * block_size - 12))
    # 两个小ROI矩形框的均值差
    feature = np.zeros(36, dtype=np.float16)
    i = 0
    fea_hist = []
    fea_diff = []
    fea_fin = []
    for re in rect:
        if re[0] < 0 or re[2] >= 320 or re[1] < 0 or re[3] >= 240:
            feature[i] = -1
        else:
            # 当前小ROI矩形框图像
            roi_img = head_img[re[1]:re[3], re[0]:re[2]].copy()
            # 计算当前小ROI直方图
            hist, bin_edge = np.histogram(roi_img.flatten())
            fea_hist.append(hist)
            # print("hist=", hist)
            diff = m - np.mean(head_img[re[1]:re[3], re[0]:re[2]])  # 中心block与当前小block的差
            feature[i] = diff
            if diff > 3:
                fea = 2
            elif diff < -3:
                fea = 1
            else:
                fea = 0
            fea_diff.append(fea)
        i += 1
        # print(i)
        cv2.rectangle(head_img, (re[0], re[1]), (re[2], re[3]), 255, 1)
    # fea_diff的计算
    hist_fin = []
    # print("fea_diff", fea_diff)
    hist_8, edge_8 = np.histogram(fea_diff[0:8], 3)
    hist_20, edge_20 = np.histogram(fea_diff[8:20], 3)
    hist_36, edge_36 = np.histogram(fea_diff[20:36], 3)
    hist_all, edge_all = np.histogram(fea_diff, 3)
    # print("hist_8=",hist_8)
    # print("hist_all=",hist_all)

    hist_fin.append(hist_8)
    hist_fin.append(hist_20)
    hist_fin.append(hist_36)
    hist_fin.append(hist_all)

    hist_fin = np.array(hist_fin)
    hist_fin = hist_fin.flatten()
    # hist_fin = hist_fin.copy().astype(np.float16)
    # print("circle_fea=", hist_fin.tolist())
    # print("len=", hist_fin.shape)

    fea_hist = np.array(fea_hist)
    fea_hist = fea_hist.flatten()
    # print("hog_fea=", fea_hist)
    # print("geature=", feature)
    # print("fea_list", feature.tolist())
    fea_con = np.concatenate((np.array(feature), np.array(hist_fin)))
    # fea_con = fea_con.tolist()
    # print("fea=", fea_con)
    fea_con = np.concatenate((fea_con, descriptor[0]))
    fea_con = np.concatenate((fea_con, fea_hist))
    fea_con = fea_con.astype(np.float16)
    fea_new = np.concatenate((feature, descriptor[0]))
    return fea_con