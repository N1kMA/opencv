import cv2 as cv
import urllib
import requests
import matplotlib.pyplot as plt

import numpy as np
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
