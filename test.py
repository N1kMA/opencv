import cv2 as cv
import urllib
import requests
import numpy as np
from datetime import datetime as dt
import datetime
import threading
import gspread
import json


class PoopyCat:
    def __init__(self,
                 kernel_open, kernel_close,
                 lower_b_m, upper_b_m,
                 lower_b_w, upper_b_w,
                 url,
                 chat_id,
                 photo_name,
                 photo_to_chat,
                 wks):

        self.kernel_open = kernel_open
        self.kernel_close = kernel_close
        self.lower_b_m = lower_b_m
        self.upper_b_m = upper_b_m
        self.lower_b_w = lower_b_w
        self.upper_b_w = upper_b_w
        self.url = url
        self.chat_id = chat_id
        self.photo_name = photo_name
        self.photo_to_chat = photo_to_chat
        self.wks = wks
        self.leaving_await()


    def video(self):
        img_response = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
        img = cv.imdecode(img_np, -1)
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        morty = self.pixel_counter(img_hsv, self.lower_b_m, self.upper_b_m)
        watson = self.pixel_counter(img_hsv, self.lower_b_w, self.upper_b_w)

        if 100000 < morty > watson:
            cat = 'morty'
        elif 100000 < watson > morty:
            cat = 'watson'
        else:
            cat = 'empty'
        return cat, img

    def leaving_await(self):
        while True:
            cat, img = self.video()
            print(cat, str(datetime.datetime.now()))
            if cat != 'empty':
                who = cat
                cv.imwrite(self.photo_name, img)
                self.send_photo(file_opened=open(self.photo_name, 'rb'))
                time_in = dt.now()
                print(f'{cat} is here! the time is {time_in}')
                while True:
                    cat, img = self.video()
                    print(cat)
                    if cat == 'empty':
                        break
                time_out = dt.now()
                wks.append_row([who, str(time_in), str(time_out)])
                print(time_out - time_in)

    def pixel_counter(self, img_hsv, lower_b, upper_b):
        mask = cv.inRange(img_hsv, lower_b, upper_b)
        mask_open = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)
        mask_close = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernel_close)
        return cv.countNonZero(mask_close)

    def send_photo(self, file_opened):
        params = {'chat_id': self.chat_id}
        files = {'photo': file_opened}
        resp = requests.post(self.photo_to_chat, params, files=files)
        return resp


if __name__ == '__main__':
    cred_file = open('creds.json', 'r')
    creds = json.loads(cred_file.read())
    sa = gspread.service_account(filename=r'service_account.json')
    sh = sa.open('kitty_poop')
    wks = sh.worksheet('data')
    photo_to_chat = creds['photo_url']
    chat_id = creds['chat_id']
    kernel_open = np.ones((5, 5))
    kernel_close = np.ones((50, 50))
    lower_b_m = np.array([0, 100, 0])
    upper_b_m = np.array([70, 255, 255])
    lower_b_w = np.array([0, 42, 0])
    upper_b_w = np.array([80, 255, 80])
    url = 'http://192.168.0.5:8080/shot.jpg'
    PoopyCat(chat_id=chat_id,
             kernel_open=kernel_open,
             kernel_close=kernel_close,
             lower_b_w=lower_b_w,
             upper_b_w=upper_b_w,
             lower_b_m=lower_b_m,
             upper_b_m=upper_b_m,
             url=url,
             wks=wks,
             photo_name='cat_photo.jpg',
             photo_to_chat=photo_to_chat)
