import cv2
import time
import numpy as np
from head_proposal import head_proposal
from fea_extract import fea_hog
from sklearn.externals import joblib
from datetime import datetime
from HeadTracker import HeadTracker
from scipy.ndimage.filters import maximum_filter, minimum_filter

## 无效数据填充
def fill_img(img):
    img.flags.writeable = True
    o = img == 65500
    u = img == 65300
    img[o] = 0
    img[u] = 0
    while np.max(u):
        img[u] = maximum_filter(img, 3)[u]
        u = np.logical_and(u, img == 65300)
    while np.max(o):
        img[o] = maximum_filter(img, 3)[o]
        o = np.logical_and(o, img == 0)
    return img

svm_persist_file = "E:\PyProjects\head_mini\model/head_v21.svm"       # with hog feature from v15
pca_persist_file = "E:\PyProjects\head_mini\model/head_v21.pca"
svc = joblib.load(svm_persist_file)
pca = joblib.load(pca_persist_file)
HT = HeadTracker()
def rt_test(img, svc, pca):
    t0 = time.time()
    # print("before filling")
    # cv2.imshow("before", img)
    img = fill_img(img)
    # cv2.imshow("filling", img)
    # cv2.waitKey(0)
    # print("after filling")
    img_u8 = (img.copy() / 20).astype(np.uint8)
    cv2.imshow("img_u8", img_u8)
    # print("111")
    # img = (img / 20).copy().reshape(240, 320).astype(np.uint8)
    # img = 120 - img
    # img = np.where(img <= 0, 0, img)
    # img = np.where(img > 100, 0, img)
    # cv2.imshow("img_src", img)
    img_src = img.copy()
    img_src = (img_src / 20).copy().reshape(240, 320)
    img_src = 140 - img_src
    # img = np.where(img > 0, img, 0)
    img_src = np.where(img_src <= 10, 0, img_src)
    # img_src = np.where(img_src > 100, 0, img_src)
    img_src = img_src.astype(np.uint8)
    img_src = cv2.GaussianBlur(img_src, (3, 3), 1.5)
    img_src = cv2.medianBlur(img_src, 3)
    # cv2.imwrite("./src/%s.png" % (datetime.now().strftime("%Y%m%d%H%M%S%f")), img_src)
    _img = img_src.copy()
    # cv2.rectangle(img_src, (280, 0), (320, 240), 0, -1)
    # for i in range(10):
    #     img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.medianBlur(img, 3)
    # ## 葛博的人头提案点
    track_img, mask = head_proposal(img, v=5)

    # 许国树的人头提案点
    # mask = head_proposal_xgs(img, img_pre)
    # img_pre = preprocess(img)
    ## 人头提案点坐标
    t1 = time.time()
    pos = np.argwhere(mask)
    # print("pos=", pos)
    # print("here")
    rect_list = []
    ## 对每一个候选点截取roi区域
    for m in pos:
        # print("idx=%s,idy=%s"%(m[0],m[1]))
        dep = img_src[m[0], m[1]]
        if m[0] < 20 or m[0] > 220 or m[1] < 20 or m[1] > 300:
            continue

        cv2.circle(_img, (m[1], m[0]), 1, 255, -1)
        # radius = (dep - 22).astype(np.uint8)              ## 0329.bin
        radius = (dep - 25).astype(np.uint8)
        # print("dep=", dep)
        print("r=", radius)
        if radius > 60:
            # print("too large...")
            radius = 60

        if radius <= 35:
            # print("too small...")
            continue
            # continue
        # print("size fitted")
        ## 框出ROI感兴趣区域矩形框
        i = m[0]
        j = m[1]
        y0 = max(0, i - radius)
        y1 = min(i + radius, 240)
        x0 = max(0, j - radius)
        x1 = min(j + radius, 320)
        # cv2.rectangle(_img, (x0, y0), (x1, y1), 255, 1)
        # print("x0=%d,x1=%d,y0=%d,y1=%d"%(x0,x1,y0,y1))
        head_img = img_src[y0:y1, x0:x1].copy()
        ## 调整大小100*100
        head_img = cv2.resize(head_img, (100, 100), interpolation=cv2.INTER_AREA)
        # head_img = cv2.blur(head_img, (3, 3))
        # 保存ROI图像作为数据集
        # if f_cnt % 2 == 0:
        cv2.imwrite("./roi/%s.png" % (datetime.now().strftime("%Y%m%d%H%M%S%f")), head_img)
        # fea = fea_extract(head_img, 6)
        fea = fea_hog(head_img, 6)
        fea = fea.flatten()
        # print("fea=", fea)
        test_x = pca.transform([fea])
        pre = svc.predict(test_x)
        if pre == "pos":
            # A[(i, j)] = 1
            ## SVM分类为人头的概率
            pro = svc.predict_proba(test_x)
            print("pro = ", pro[0])
            proba = max(pro[0])
            ## 分类置信率阈值
            if proba >= 0.6:
                rect_list.append([x0, y0, x1, y1, i, j, proba])
                # print("pos=",[x0,y0,x1,y1,i,j,proba])
                # cv2.rectangle(_img, (x0, y0), (x1, y1), 255, 1)
                # print("probability=", proba)
    # print("rect_list=", len(rect_list))
    dele_idx = []
    num = len(rect_list)
    if num > 1:
        for i in range(num - 1):
            for j in range(i + 1, num):
                # print("i=%s, j=%s"%(i,j))
                # print("i=",rect_list[i])
                # print("j=",rect_list[j])
                overlap = max(rect_list[j][0], rect_list[i][0]) - min(rect_list[j][2], rect_list[i][2])
                # print("guole")
                ## 如果有重叠
                if overlap < -30:
                    if rect_list[i][-1] > rect_list[j][-1]:
                        dele_idx.append(j)
                    else:
                        dele_idx.append(i)
    ## NMS后的人头结果降序排列并删除
    dele_idx = list(set(dele_idx))
    dele_idx.sort(reverse=True)
    # print("dele idx=", dele_idx)
    for idx in dele_idx:
        del rect_list[idx]
    ## 当前帧的人头 mask
    A = np.zeros(240 * 320, dtype=np.bool)
    A = A.reshape(240, 320)
    for r in rect_list:
        cv2.circle(_img, (r[5], r[4]), 40, 255, 1)
        cv2.rectangle(_img, (r[0], r[1]), (r[2], r[3]), 255, 1)
        A[int(r[4]), int(r[5])] = 1
    print("time=", (time.time() - t1))
    # cv2.putText(_img, "human detection", (5,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)
    # ## 保存npz文件img和mask
    # pre_img.append(img_u8)
    # human_mask.append(A)
    # cv2.rectangle(_img, (20, 20), (300, 220), 255, 1)

    ## 绘制轨迹
    HT.update(track_img, A)
    for j in HT.get_valid_tracks():
        print(j.omission_cnt, j.path)
        pathLen = len(j.path)
        for w in range(1, pathLen):
            cv2.line(_img, j.path[w - 1][::-1], j.path[w][::-1], (255, 0, 0), 2)
            # pathLen = len(j.pre_path)
            # for w in range(1, pathLen):
            #     cv2.line(_img, j.pre_path[w - 1][::-1], j.pre_path[w][::-1], (255, 0, 0), 2)

    ## 人头跟踪结果
    cv2.putText(_img, "#OUT:%s" % HT.inn, (10, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255, 1)
    cv2.putText(_img, "#IN:%s" % HT.out, (180, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255, 1)
    cv2.putText(_img, "@SHUJI 2018", (200, 230), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, 255, 1)
    cv2.imshow("rec", _img)
    t = time.time() - t0
    print("processing time = ", t)
    cv2.waitKey(1)