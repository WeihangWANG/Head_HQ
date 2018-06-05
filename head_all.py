#============================================================#
############# SHUJI HEAD DETECTION Version 1.0 ###############
#============================================================#
#==========================2018-04-17========================#


import time
import cv2
import numpy as np
from HeadTracker import HeadTracker
from head_proposal import head_proposal
from head_verify import head_verify
from datetime import datetime

f_cnt = 0
#mouse callback function
def draw_circle(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDBLCLK:
        dep = img_u8[y,x]
        print("mouse=", dep)
        cv2.circle(img_u8, (x,y), 2, 255, -1)
        radius = int(140 * 400 / dep / 20)
        y0 = max(0, y - radius - 5)
        y1 = min(y + radius + 5, 240)
        x0 = max(0, x - radius - 5)
        x1 = min(x + radius + 5, 320)
        cv2.rectangle(img_u8, (x0, y0), (x1, y1), 255, 1)
        cv2.circle(img_u8, (x, y), radius, 255, 1)
        cv2.imshow("img_u8", img_u8)

## 初始化人头跟踪子类
HT = HeadTracker()

file_dep = open("E:\PyProjects\head_mini_offline/20180416.bin", 'rb')
# file_dep = open("H:/20180511.bin", 'rb')
# file_dep = open("D:\TOF\HEAD\data/20180525.bin", 'rb')
## 从f_cnt帧开始读取
img = np.frombuffer(file_dep.read(2 * 320 * 240 * f_cnt), dtype=np.uint16)

t_roi = 0
t_cls = 0
t_trc = 0
t_tol = 0
## 主程序入口
while True:
# while f_cnt < 1800:
    img = np.frombuffer(file_dep.read(2 * 320 * 240), dtype=np.uint16).reshape(240, 320)
#     img = np.frombuffer(file_dep.read(8 * 320 * 240), dtype=np.float64).reshape(240, 320)
#     img = img.astype(np.uint16)
    img_u8 = (img/20).astype(np.uint8)
    ## 鼠标事件
    cv2.namedWindow('img_u8')
    cv2.setMouseCallback('img_u8', draw_circle)
    cv2.imshow("img_u8", img_u8)
    # print("frame_num = ", f_cnt)
    # cv2.imwrite("./src_u8/%s.png"%(f_cnt), img_u8)
    f_cnt = f_cnt + 1
    print("f_cnt = ", f_cnt)
    t0 = time.time()
    ## 人头提案点检测
    track_img, mask = head_proposal(img, v=5)
    img_src = track_img.copy()
    # cv2.imshow("src", img_src)
    t1 = time.time()
    pro_time = t1 - t0
    ## 人头ROI特征提取与分类
    _img, A = head_verify(img_src, mask)
    t2 = time.time()
    reg_time = t2 - t1
    ## 人头跟踪与轨迹绘制
    HT.update(track_img, A)
    for j in HT.get_valid_tracks():
        print(j.omission_cnt,j.path)
        pathLen = len(j.path)
        for w in range(1, pathLen):
            cv2.line(_img, j.path[w - 1][::-1], j.path[w][::-1], (255, 0, 0), 2)

    track_time = time.time() - t2
    total_time = time.time() - t0

    _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2BGR)
    # _img = cv2.applyColorMap(_img, cv2.COLORMAP_SPRING)
    cv2.putText(_img, "#OUT: %s" % HT.inn, (10, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
    cv2.putText(_img, "#IN: %s" % HT.out, (180, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
    t_roi = t_roi + int(pro_time*1000)
    t_cls = t_cls + int(reg_time*1000)
    t_trc = t_trc + int(track_time*1000)
    t_tol = t_tol + int(total_time*1000)
    # cv2.imwrite("./track/%s.png"%(datetime.now().strftime("%Y%m%d%H%M%S%f")), _img)
    cv2.putText(_img, "@SJTU 2018", (200, 230), cv2.FONT_HERSHEY_COMPLEX_SMALL+cv2.FONT_ITALIC, 0.7, (255,255,0), 1)
    # cv2.putText(_img, "pro_t=%s" % (int(pro_time * 1000)), (10, 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,0), 1)
    # cv2.putText(_img, "reg_t=%s"%(int(reg_time*1000)), (10,190), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,0), 1)
    # cv2.putText(_img, "trk_t=%s"%(int(track_time*1000)), (10,210), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,0), 1)
    # cv2.putText(_img, "tol_t=%s"%(int(total_time*1000)), (10,230), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,0), 1)
    cv2.imshow("rec", _img)

    cv2.waitKey(0)
print(t_roi/f_cnt, t_cls/f_cnt, t_trc/f_cnt, t_tol/f_cnt)

# file_sav.close()
# human_mask =np.array(human_mask,dtype=np.bool)
# print(human_mask.shape)
# np.savez_compressed("rec_mask.npz", mask=human_mask)
# np.savez("head_det.npz", img=pre_img, mask=human_mask)