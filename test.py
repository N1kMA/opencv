import cv2 as cv
import urllib
import requests
import matplotlib.pyplot as plt
import numpy as np


def video_img():
    url = 'http://192.168.0.7:8080/shot.jpg'
    while True:
        img_response = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
        img = cv.imdecode(img_np, -1)
        cv.imshow('test', img)
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        plt.figure()
        plt.title('Grayscale Histogram')
        plt.xlabel('Bins')
        plt.ylabel('# of pixels')
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
        if cv.waitKey(5) & 0xFF == ord('d'):
            break


def img():
    lower_b = np.array([45, 55, 100])
    upper_b = np.array([85, 255, 255])
    img = cv.imread(r'C:\Users\rrapi\Pictures\sprite.jpg')
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, lower_b, upper_b)
    res = cv.bitwise_and(img, img, mask=mask)
    print(cv.countNonZero(mask))
    cv.imshow('mask', mask)
    cv.imshow('hsv', img_hsv)
    cv.imshow('res', res)
    cv.imshow('bgr', img)
    cv.waitKey(0)


if __name__ == '__main__':
    img()