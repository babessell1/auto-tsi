from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd

def find_nearest(lst, val):
    lst = np.asarray(lst)
    idx = (np.abs(lst - val)).argmin()
    return idx, lst[idx]

def match_images(treeDic, imageTimeFile):
    # match images taken by the camera that were taken at the same time as the
    # lidar detected a round object while the camera is in dead center

    # treeDic is a pandas dataframe or numpy array
    imageTimes = []
    handle = open(imageTimeFile, 'r')
    for line in handle:
        thisLine = line.strip()
        if thisLine.startswith("15"):
            imageTimes.append(thisLine)
    imageTimes = np.asarray(imageTimes).astype(np.int64)

    treeDic_inView = treeDic[treeDic.inView==True]
    #for index, row in treeDic_inView.iterrows():
        #index, t  = index, int(row['Timestamp'])
    dfsize = len(treeDic.index)
    imgNameLst = [None]*dfsize
    #idxLst = np.empty(dfsize, dtype=int)
    for index, row in treeDic.iterrows():
        if row['inView'] == 'TRUE' or row['inView'] == True:
            t = int(row['Timestamp'])
            tidx, time = find_nearest(imageTimes, t)
            print(index)
            imgNameLst[int(index)] = 'img_' + str(tidx)

    treeDic['imgName'] = pd.Series(imgNameLst, index=treeDic.index)

    return treeDic

def load_yolo(weighting_file, cfg_file, naming_file):
	cnnet = cv2.dnn.readcnnet(weighting_file, cfg_file)
	class_list = []
	with open(naming_file, "r") as f:
		class_list = [line.strip() for line in f.readlines()]
	layer_name = cnnet.getLayerNames()
	layer_outputs = [layer_name[i[0]-1] for i in cnnet.getUnconnectedOutLayers()]
	color_list = np.random.uniform(0, 255, size=(len(class_list), 3))
	return cnnet, class_list, color_list, layer_outputs

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	#img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

def object_detection(img, cnnet, outputLayers):
	countour = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	cnnet.setInput(contour)
	outputs = cnnet.forward(outputLayers)
	return contour, outputs

def retrieve_box_dimensions(outputs, height, width):
	boxs = []
	confidence_score_list = []
	classID = []
	for output in outputs:
		for out in output:
			score_list = out[5:]
			class_id = np.argmax(score_list)
			confidence_score = score_list[class_id]
			if confidence_score > 0.3:
				x_center = int(detect[0] * width)
				y_center = int(detect[1] * height)
				width = int(detect[2] * width)
				height = int(detect[3] * height)
				x = int(x_center - width/2)
				y = int(y_center - height / 2)
				boxs.append([x, y, width, height])
				confidence_score_list.append(float(confidence_score))
				classID.append(class_id)
	return boxs, confidence_score_list, classID

def label_draw(boxs, confidence_score_list, color_list, classID, class_list, img):
	indexes = cv2.dnn.NMSboxs(boxs, confidence_score_list, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxs)):
		if i in indexes:
			x, y, w, h = boxs[i]
			label = str(class_list[classID[i]])
			color = color_list[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imshow("Image", img)

def cam_calibration():
    # This function was directly taken and modified from:
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('calibrationImages/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return ret, mtx, dist, rvecs, tvecs

ret, mtx, dist, rvecs, tvecs = cam_calibration()
print(mtx, dist)

def calc_height(dist, boxheight):
    # dist - distance to tree from lidar sensor (mm)
    # box_height - height of bounding box from yolo object detection (px)
    height_sensor = 1372 # mm (4.5 ft)
    focal_length = 3.04 # mm
    height_image = 1080 # px
    height_tree = dist*box_height*height_sensor/(focal_length*height_image)

    return height_tree
