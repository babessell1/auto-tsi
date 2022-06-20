from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

red = (255, 0, 0)

def show(image):
    plt.figure(figsize=(10,10))
    plt.imshow(image,interpolation='nearest')

def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    image = image.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def circle_contour(image, contour):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    treeHeight = max(ellipse[1][:])
    print(treeHeight)

    cv2.ellipse(image_with_ellipse, ellipse, red, 2, cv2.LINE_AA)
    return image_with_ellipse, treeHeight


def findTree(image):
    # reverse color memory storage
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # scale image
    max_dimension = max(image.shape)
    scale = 700/max_dimension
    image = cv2.resize(image, None, fx=scale, fy=scale)

    # filter noise
    image_blur = cv2.GaussianBlur(image, (7,7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    # color filter
    min_color = np.array([20,10,10])
    max_color = np.array([30,30,40])

    mask = cv2.inRange(image_blur_hsv, min_color, max_color)

    # seperate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    biggest_tree_contour, mask_trees = find_biggest_contour(mask_clean)

    overlay = overlay_mask(mask_clean, image)

    circled, ellipse = circle_contour(overlay, biggest_tree_contour)
    show(circled)

    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)

    return bgr
