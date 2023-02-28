import cv2
import numpy as np

def Barcode_Detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    s_kernel = [[0,-1,0],[-1,5,-1],[0,-1,0]]
    sharp_A = np.squeeze(np.asarray(s_kernel))
    sharp = cv2.filter2D(gray, -1, sharp_A)
    #deNoise = cv2.fastNlMeansDenoisingColored(sharp,None,5,10,7,21)
    cv2.imshow("Processed", sharp)
    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,7.5) # Adaptive Thresholding with Gaussian Blur and Binary Threshold
    kernel = [[-2,1,-1],[1,6,1],[-1,1,-2]]
    A = np.squeeze(np.asarray(kernel))
    dst = cv2.filter2D(th3, -1, A)
    kernel = np.ones((1,1),np.uint8)
    img_dilation = cv2.dilate(dst, kernel, iterations=3)
    added = cv2.add(th3,img_dilation)
    median = cv2.medianBlur(added, 3)
    w, l = median.shape
    crop = median[(w//7):(6*w//7), (l//9):8*(l//9)]
    cv2.imshow("Final", crop)

img = cv2.imread('BarcodeDetection\Images\ProductBarcode008.jpg')
cv2.imshow('Original', img)
Barcode_Detector(img)

cv2.waitKey(0)