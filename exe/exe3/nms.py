import numpy as np
from matplotlib import pyplot as plt
import numpy as np

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        print(ovr)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

dets = np.array([[65,65,65+50,65+50, 0.4], [125,120,125+25,120+30, 0.6], [60,60,60+60,60+60, 0.9], [120,120,120+30,120+30, 0.7]])
print(nms(dets, 0.7))
