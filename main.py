import numpy as np
import cv2
from PIL import Image
from mss import mss
from screeninfo import get_monitors

for m in get_monitors():
    print(str(m))

bounding_box = {
    'top': (m.height-(m.height/1.5))/2, 
    'left': 0, 
    'width': m.width/2, 
    'height': m.height/1.5
}
sct = mss()

windowWidth = int(m.width/4)
windowHeight = int(m.height/3)
windowOffset = 0

window = "pyTetris"
cv2.namedWindow(window)
cv2.moveWindow(window, m.width-windowWidth-windowOffset, m.height-windowHeight-55-windowOffset)

template = cv2.imread('assets/4x1.png', -1)
h, w = template.shape[:-1]

template2 = cv2.imread('assets/1x4.png', -1)
h2, w2 = template2.shape[:-1]

template3 = cv2.imread('assets/2x2greenH.png', -1)
h3, w3 = template3.shape[:-1]

threshold = 0.99

while True:
    sct_img = sct.grab(bounding_box)
    img = np.array(sct_img)
    img = cv2.resize(img, (windowWidth, windowHeight))
    
    # Match 4x1
    match = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    loc = np.where( match >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (244, 220, 0), 2)
        
    # Match 1x4
    match = cv2.matchTemplate(img, template2, cv2.TM_CCORR_NORMED)
    loc = np.where( match >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w2, pt[1] + h2), (244, 220, 0), 2)
    
    # Match 2x2green
    match = cv2.matchTemplate(img, template3, cv2.TM_CCORR_NORMED)
    loc = np.where( match >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w3, pt[1] + h3), (244, 220, 0), 2)
    
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    
    # location = max_loc
    # bottom_right = (location[0] + w, location[1] + h)
    
    # cv2.rectangle(img, location, bottom_right, 255, 5)
    
    
    cv2.imshow(window, img)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break