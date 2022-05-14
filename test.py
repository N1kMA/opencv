import cv2 as cv
import urllib
import requests
import numpy as np
from datetime import datetime as dt
import datetime
import threading
import gspread
sa = gspread.service_account(filename=r'D:\Python\google_sheets\service_account.json')
sh = sa.open('kitty_poop')
wks = sh.worksheet('data')


kernel_open = np.ones((5, 5))
kernel_close = np.ones((50, 50))
lower_b_m = np.array([0, 100, 0])
upper_b_m = np.array([70, 255, 255])
lower_b_w = np.array([0, 42, 0])
upper_b_w = np.array([80, 255, 80])



url = 'http://192.168.0.6:8080/shot.jpg'


def video():
    img_response = urllib.request.urlopen(url)
    img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
    img = cv.imdecode(img_np, -1)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    morty = pixel_counter(img_hsv, lower_b_m, upper_b_m)
    watson = pixel_counter(img_hsv, lower_b_w, upper_b_w)

    if 100000 < morty > watson:
        cat = 'morty'
    elif 100000 < watson > morty:
        cat = 'watson'
    else:
        cat = 'empty'
    return cat, img


def leaving_await():
    while True:
        cat, img = video()
        print(cat)
        if cat != 'empty':
            who = cat
            time_in = dt.now()
            print(f'{cat} is here! the time is {time_in}')
            while True:
                cat, img = video()
                print(cat)
                if cat == 'empty':
                    break
            time_out = dt.now()
            wks.append_row([who, str(time_in), str(time_out)])
            print(time_out - time_in)



def pixel_counter(img_hsv, lower_b, upper_b):
    mask = cv.inRange(img_hsv, lower_b, upper_b)
    mask_open = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)
    mask_close = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernel_close)
    return cv.countNonZero(mask_close)


if __name__ == '__main__':
    leaving_await()
