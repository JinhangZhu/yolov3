# Libraries
# import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


try:
    from utils.utils import plot_one_box
except ImportError:
    from .utils.utils import plot_one_box


def diff_hands(img, boxes, open_size=5, extend_scale=0.5, bin_thres=100, view=False):
    """Differentiate the predited bounding boxes into two labels. <left_hand>, <right_hand>
    Arguments:
        img: image
        boxes: predicted bounding boxes, format: [(ID, conf, centre_x, centre_y, width, height)]
    Return:
        diff_boxes: differentiated bounding boxes, format: (ID, centre_x, centre_y, width, height), where ID relates to new 'labels'.
    """
    # Info
    if isinstance(boxes, list):
        boxes = np.asarray(boxes)
    n_hands = boxes.shape[0]

    if n_hands == 0:
        print("No hands detected.")
        return 0

    elif n_hands == 1:
        # Check extended boxes
        ellipse_info = extractHandArm(img, boxes, open_size=open_size,
                                      extend_scale=extend_scale, view=view)
        angle = ellipse_info[0][1]
        boxes[0][0] = identifyOneHand(angle)

    elif n_hands == 2:
        # Check a-little-bit extended boxes
        ellipse_info = extractHandArm(img, boxes, open_size=open_size,
                                      extend_scale=extend_scale, view=view)
        _, angle1 = ellipse_info[0]
        _, angle2 = ellipse_info[1]

        # Flag that angles are 0-90 and 90-180 separately.
        isAngleRight1 = identifyOneHand(angle1)
        isAngleRight2 = identifyOneHand(angle2)
        isAngleDiff = isAngleRight1 ^ isAngleRight2

        if not isAngleDiff:
            c_x0 = (boxes[0][1] + boxes[0][3]) / 2  # center x0
            c_x1 = (boxes[1][1] + boxes[1][3]) / 2  # center x1
            boxes[0][0] = int(c_x0 > c_x1)
            boxes[1][0] = int(c_x0 < c_x1)
        else:
            boxes[0][0] = isAngleRight1
            boxes[1][0] = isAngleRight2

    else:   # More hands detected
        # Sort the bounding boxes according to the confidences
        boxes = boxes[boxes[:, 1].argsort()[::-1]]  # Descending order
        boxes = boxes[:2, :]
        boxes = diff_hands(img, boxes)   # Recursive step for once

    return boxes


def display(log, save_path):
    for i, l in enumerate(log):
        plt.figure(figsize=(20, 10), dpi=300)
        plt.subplot(141)
        plt.imshow(l[0])
        plt.axis('off')
        # plt.title('Original')
        plt.subplot(142)
        plt.imshow(l[1])
        plt.axis('off')
        # plt.title('Thresholding')
        plt.subplot(143)
        plt.imshow(l[2])
        plt.axis('off')
        # plt.title('Open Operation')
        plt.subplot(144)
        plt.imshow(l[3])
        plt.axis('off')
        # plt.title('Ellipse fitting')
        plt.savefig(save_path + '_out_'+'%d' % i + '.jpg')
        plt.show()


def identifyOneHand(angle):
    """Identify one hand as left/right hand.
    Arguments:
        #shape: image.shape
        #pos: (x, y)
        angle: degree
    Returns:
        0 for left_hand, 1 for right_hand
    """
    # x, y = pos      # positions
    # w = shape[1]    # Width
    # ct_axis = w/2   # center axis

    if angle <= 90:
        return 0
    else:
        return 1


def extractHandArm(img, boxes, open_size=3, extend_scale=0, view=False):
    """Extract hands and arms around them from bounding boxes and the image.
    Arguments:
        img: numpy array
        boxes: list of numpy arrays, format: (cls, conf, ltx, lty, rbx, rby)
    Returns:
        res: [((x, y), angle), ]
    """
    crops = crop_bbox(img, boxes)
    exd_crops, exd_boxes = extendRegion(img, boxes, extend_scale)

    # Returns
    res = []

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(crops))]

    for i, (crop, exd_crop) in enumerate(zip(crops, exd_crops)):
        patch = exd_crop
        # Denoising
        patch = cv2.blur(patch, (3, 3))  # Mean-value filter
        patch = cv2.GaussianBlur(patch, (3, 3), 0)  # Gaussian filter
        patch = cv2.medianBlur(patch, 5)    # Median filter

        # Morphorlogy: open operation
        patch = openOperation(patch, open_size)

        # YCrCb colorspace thresholding and segmentation
        r, _ = skinMask(patch, colors, i, view)
        res.append(r)

    return res


# Method： YCrCb colorspace Cr-OTSU thresholding
# References: https://blog.csdn.net/qq_41562704/article/details/88975569


def skinMask(region, colors, i, view=False):
    if view:
        to_subplots = [region]

    YCrCb = cv2.cvtColor(region, cv2.COLOR_RGB2YCrCb)  # 转换至YCrCb空间
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu处理
    thresholded = cv2.bitwise_and(region, region, mask=skin)
    thresholded_copy = thresholded.copy()

    if view:
        to_subplots.append(thresholded)

    contours, hierarchy = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(thresholded_copy, [c], -1, [255, 255, 255], 3)

        if view:
            to_subplots.append(thresholded_copy)

        # draw the biggest contour (c) in green
        ellipse = cv2.fitEllipse(c)
        cv2.ellipse(thresholded_copy, ellipse, colors[i], 2)

        # Plotting config
        tl = round(0.004 * (thresholded_copy.shape[0] + thresholded_copy.shape[1]) / 2) + 1  # Line thickness
        tf = max(tl-1, 1)   # Font thickness
        rd = tl

        x, y, angle = int(ellipse[0][0]), int(ellipse[0][1]), int(ellipse[2])
        angle = rectifyAngle(angle)
        pose = ((x, y), angle)
        label_pos = '({}, {})'.format(x, y)
        label_ang = '{} deg'.format(angle)

        # Annotation
        ellipsed = thresholded_copy.copy()
        cv2.circle(ellipsed, (x, y), rd, colors[i], -11)
        cv2.putText(ellipsed, label_pos, (x - 70, y - 20), 0, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(ellipsed, label_ang, (x - 70, y + 35), 0, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        if view:
            to_subplots.append(ellipsed)
            plt.figure(figsize=(20, 10), dpi=300)
            ls = len(to_subplots)
            for i, sp in enumerate(to_subplots):
                plt.subplot(1, ls, i+1)
                plt.imshow(sp)
                plt.axis('off')
            plt.show()

    return pose, thresholded

# 采用KMeans clustering获取手部


def kmeansMask(region, k=3):
    """KMeans Clustering on the region to get the hand.

    Reference: https://morioh.com/p/b6763f7527d5

    Returns:
        masked_region:
        avg_color:
    """
    # 将3D的图片reshape为2D的array，即宽高二维上压平为一维
    pixel_values = region.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    # print('After flattening: ', pixel_values.shape)

    # 停止策略
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # kmeans clustering
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 坐标类型恢复
    centers = np.uint8(centers)

    segmented_region = centers[labels.flatten()]  # 配对对应label
    segmented_region = segmented_region.reshape(region.shape)  # 恢复size
    # plt.imshow(segmented_region)
    # plt.show()

    # 找到最大的cluster，也就是出现次数最多的label
    most_label = np.bincount(labels.flatten()).argmax()
    # print('Most frequently: %s' % most_label)

    # 选择性显示/掩膜显示
    # colors = [[random.randint(200, 255) for _ in range(3)] for _ in range(3)]

    masked_region = np.copy(region)
    masked_region = masked_region.reshape((-1, 3))
    mask = labels == most_label
    # masked_region[mask[:, 0], :] = colors[1]  # 最大的区域染色
    masked_region[~mask[:, 0], :] = [0, 0, 0]  # 其余统一为黑色
    masked_region = masked_region.reshape(region.shape)
    # plt.imshow(masked_region)
    # plt.show()

    # 找到手的平均颜色（即cluster的平均颜色）
    most_mask = labels == most_label
    most_avg = np.mean(pixel_values[most_mask[:, 0], :], axis=0, dtype=np.int32)
    # color_demo = np.tile(most_avg, 10000).reshape((100, 100, 3))

    return masked_region, most_avg


def openOperation(region, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(region, kernel)
    dilation = cv2.dilate(erosion, kernel)
    return dilation


def extendRegion(img, boxes, extend_scale=1):
    if isinstance(boxes, list):
        boxes = np.asarray(boxes, dtype=np.float32)
    xyxy = boxes[:, 2:].astype(int)
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    exd_x0 = np.clip(xyxy[:, 0] - (w*extend_scale).astype(int), 0, img.shape[1])
    exd_y0 = np.clip(xyxy[:, 1] - (h*extend_scale).astype(int), 0, img.shape[0])
    exd_x1 = np.clip(xyxy[:, 2] + (w*extend_scale).astype(int), 0, img.shape[1])
    exd_y1 = np.clip(xyxy[:, 3] + (h*extend_scale).astype(int), 0, img.shape[0])

    exd_crops = []
    for i in range(boxes.shape[0]):
        crop_img = img[exd_y0[i]:exd_y1[i], exd_x0[i]:exd_x1[i]]
        exd_crops.append(crop_img)

    exd_boxes = np.concatenate(
        (boxes[:, :2], exd_x0.reshape((-1, 1)), exd_y0.reshape((-1, 1)),
         exd_x1.reshape((-1, 1)), exd_y1.reshape((-1, 1))),
        axis=1
    )

    return exd_crops, exd_boxes


def colorMask(region, color, thres=50):
    flatten_region = (region - color).reshape((-1, 3))
    near_mask = np.linalg.norm(flatten_region, axis=1) < thres

    masked_region = np.copy(region)
    masked_region = masked_region.reshape((-1, 3))
    masked_region[~near_mask, :] = [0, 0, 0]
    masked_region = masked_region.reshape(region.shape)

    return masked_region


def crop_bbox(img, boxes, raw=False):
    if raw:
        xyxy = boxes[:, 1:].astype(int)
    else:
        xyxy = boxes[:, 2:].astype(int)
    crops = []
    for i in range(boxes.shape[0]):
        crop_img = img[xyxy[i, 1]:xyxy[i, 3], xyxy[i, 0]:xyxy[i, 2]]
        crops.append(crop_img)

    return crops


def boxes_info(boxes, labels):
    for i, box in enumerate(boxes):
        label = labels[int(box[0])]
        # conf = box[1]
        print(
            '\nNormalised bounding box {}: {}\n'.format(i, label),
            '%10s' * 5 % ('conf', 'LT_x', 'LT_y', 'RB_x', 'RB_y'),
            '%10.4s' * 5 % (*box, )
        )


def get_pred_boxes(path):
    """Get boxes from predictions: cls_id, conf, xmin, ymin, xmax, ymax
    """
    boxes = []
    with open(path, 'r') as f:
        for line in f:
            line = line[:-1].split()
            line[0] = int(line[0])
            line[1:] = [float(i) for i in line[1:]]
            boxes.append(line)
    return boxes


def get_ann_boxes(path):
    """Get boxes from the annotation of one image: cls_id, cx, cy, w, h
    """
    boxes = []
    with open(path, 'r') as f:
        for line in f:
            line = line[:-1].split()
            line[0] = int(line[0])  # cls id
            line[1:] = [float(i) for i in line[1:]]  # x y w h
            boxes.append(line)
    return boxes


def scale_xyxy(x, img_shape, to_image=True):
    """Rescale to image shape or normalize from image shape.
    """
    if to_image:
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0::2] = x[:, 0::2] * img_shape[1]  # x
        y[:, 1::2] = x[:, 1::2] * img_shape[0]  # y
    else:
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0::2] = x[:, 0::2] / img_shape[1]  # x
        y[:, 1::2] = x[:, 1::2] / img_shape[0]  # y
    return y


def rectifyAngle(angle):
    """Rectify angle from cv2 numpy array to intuitions.
    """
    return -angle+90+180*(angle > 90)


def write_result(boxes, img, labels, save_path=None):
    copy = img.copy()
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(labels))]
    # boxes = scale_coords(torch.Size([320, 512]), boxes, img.shape).round()
    for box in boxes:
        label = '%s %.2f' % (labels[int(box[0])], box[1])
        plot_one_box(box[2:], copy, label=label, color=colors[int(box[0])])
    # cv2.imwrite(save_path, img)
    return copy


def isOverlap1D(box1, box2):
    """Check if two 1D boxes overlap.
    Reference: https://stackoverflow.com/a/20925869/12646778
    Arguments:
        box1, box2: format: (xmin, xmax)
    Returns:
        res: bool, True for overlapping, False for not
    """
    xmin1, xmax1 = box1
    xmin2, xmax2 = box2
    return xmax1 >= xmin2 and xmax2 >= xmin1


def isOverlap2D(box1, box2):
    """Check if the two 2D boxes overlap.
    Arguments:
        box1, box2: format: (ltx, lty, rbx, rby)
    Returns:
        res: bool, True for overlapping, False for not
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    return isOverlap1D((xmin1, xmax1), (xmin2, xmax2)) and isOverlap1D((ymin1, ymax1), (ymin2, ymax2))
