import cv2 as cv
import urllib
import requests
import matplotlib.pyplot as plt
import numpy as np
kernel_open = np.ones((3, 3))
kernel_close = np.ones((50, 50))
#lower_b = np.array([17, 20, 45])
#upper_b = np.array([50, 100, 240])
lower_b = np.array([0, 0, 0])
upper_b = np.array([0, 0, 80])
url = 'http://192.168.0.5:8080/shot.jpg'


def video_img():

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


def video():
    while True:
        img_response = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
        img = cv.imdecode(img_np, -1)
        #img = cv.imread(r'C:\Users\rrapi\Pictures\cat.jpg')
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img_hsv, lower_b, upper_b)
        mask_open = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)
        mask_close = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernel_close)
        res = cv.bitwise_and(img, img, mask=mask)
        conts, h = cv.findContours(mask_close.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # cv.drawContours(img, conts, -1, (0, 255, 255), 3)
        for i in range(len(conts)):
            x, y, w, h = cv.boundingRect(conts[i])
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        print(cv.countNonZero(mask))
        #cv.imshow('mask', mask)
        #cv.imshow('mask_open', mask_open)
        cv.imshow('mask_close', mask_close)
        # cv.imshow('hsv', img_hsv)
        #cv.imshow('res', res)
        cv.imshow('bgr', img)
        if cv.waitKey(5) & 0xFF == ord('d'):
            break


def img():
    lower_b = np.array([17, 20, 45])
    upper_b = np.array([50, 100, 240])
    img = cv.imread(r'C:\Users\rrapi\Pictures\cat.jpg')
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, lower_b, upper_b)
    mask_open = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)
    mask_close = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernel_close)
    res = cv.bitwise_and(img, img, mask=mask)
    conts, h =cv.findContours(mask_close.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #cv.drawContours(img, conts, -1, (0, 255, 255), 3)
    for i in range(len(conts)):
        x, y, w, h = cv.boundingRect(conts[i])
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
    print(cv.countNonZero(mask))
    cv.imshow('mask', mask)
    cv.imshow('mask_open', mask_open)
    cv.imshow('mask_close', mask_close)
    #cv.imshow('hsv', img_hsv)
    cv.imshow('res', res)
    cv.imshow('bgr', img)
    cv.waitKey(0)




if __name__ == '__main__':
    video()