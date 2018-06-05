#============================================================#
############# SHUJI HEAD DETECTION Version 1.0 ###############
#============================================================#
#==========================2018-04-17========================#


from core import *
import time


mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

params = cv2.SimpleBlobDetector_Params()
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = False
params.filterByCircularity = False
params.filterByArea = False
params.filterByCircularity = True
params.minCircularity = 0.01
params.filterByArea = True
# params.minThreshold = 10
# params.maxThreshold = 200
params.minDistBetweenBlobs = 60
params.minThreshold = 10
params.maxThreshold = 100
params.thresholdStep = 1
params.minArea = 1000.0
params.maxArea = 8000.0
detector = cv2.SimpleBlobDetector_create(params)


def head_proposal(img, v=3):
    if v==1:
        local_maxima = find_local_maxima(img, 4)
        #plot_mask('local_maxima1', img, local_maxima)
        local_maxima = clustering(local_maxima, 5)
        #plot_mask('local_maxima2', img, local_maxima)
        meanshift = mean_shift(img, local_maxima, 2)
        #plot_mask('meanshift', img, meanshift)

    elif v==3:
        # best param: 55, 15
        # evaluation_result(recall=0.7129629629629629, nr=0.4310067358062919, cnt=9.865384615384615)

        # best param: 55, 25
        # evaluation_result(recall=0.6997455470737913, nr=0.38551526468287806, cnt=11.012722646310433)

        # best param: 55, 30
        # evaluation_result(recall=0.6972010178117048, nr=0.3768135766129698, cnt=8.017811704834605)

        # best param: 55, 35
        # evaluation_result(recall=0.6972010178117048, nr=0.3691404544365296, cnt=6.659033078880407)
        _img = preprocess(img)
        img = morph_open(_img, 55)
        local_maxima = find_local_maxima(img, 35)
        local_maxima = cluster(local_maxima)
        return _img, local_maxima

    elif v==4:
        # best param: 55, 15
        # evaluation_result(recall=0.7129629629629629, nr=0.4310067358062919, cnt=9.865384615384615)

        # best param: 55, 25
        # evaluation_result(recall=0.6997455470737913, nr=0.38551526468287806, cnt=11.012722646310433)

        # best param: 55, 30
        # evaluation_result(recall=0.6972010178117048, nr=0.3768135766129698, cnt=8.017811704834605)

        # best param: 55, 35
        # evaluation_result(recall=0.6972010178117048, nr=0.3691404544365296, cnt=6.659033078880407)
        img = preprocess(img)
        fgmask = mog.apply(img)
        dilated = cv2.dilate(fgmask, np.ones((3,3), dtype=np.uint8), iterations=2)
        image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fgmask = np.zeros_like(fgmask, dtype=np.bool)
        for c in contours:
            if cv2.contourArea(c)>1600:
                (x,y,w,h) = cv2.boundingRect(c)
                fgmask[y:y+h,x:x+w] = True
        img_ = np.where(fgmask, img, 0)
        # img__ = morph_open(img_, 55)
        local_maxima = find_local_maxima(np.where(fgmask, img_, 255), 25)
        # local_maxima = cluster(local_maxima)
        local_maxima = shape_center(img_, local_maxima)
        return img, local_maxima

    elif v==5:
        # start = time.time()
        img = preprocess(img)
        # print('  preprocess costs %d ms' % ((time.time() - start) * 1000))
        # start = time.time()
        fgmask = mog.apply(img)
        dilated = cv2.dilate(fgmask, np.ones((3,3), dtype=np.uint8), iterations=2)
        image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fgmask = np.zeros_like(fgmask, dtype=np.bool)
        for c in contours:
            if cv2.contourArea(c)>1600:
                (x,y,w,h) = cv2.boundingRect(c)
                fgmask[y:y+h,x:x+w] = True
        img_ = np.where(fgmask, img, 0)
        # print('  foreground extract costs %d ms' % ((time.time() - start) * 1000))
        # start = time.time()
        _img = mean_pooling(img_, 10).astype(np.uint8)
        _img = cv2.resize(_img, (320, 240))

        keypoints = detector.detect(_img)
        trace_mask = np.zeros_like(img)
        for hp in keypoints:
            trace_mask[int(hp.pt[1]),int(hp.pt[0])] = True
        # print('  blob detect costs %d ms' % ((time.time() - start) * 1000))

        return img, trace_mask



def region_restrict(mask,n=20):
    mask[:n, :] = False
    mask[-n:, :] = False
    mask[:, :n] = False
    mask[:, -n:] = False
    return mask