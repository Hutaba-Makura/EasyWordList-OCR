##   Show a color histgram of selected pixels
##   Usage:
##       $ python3 color_hist.py image_file

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2


def mouse_event(event, x, y, flags, param):

    global points, img, img_red, src, masks, radius, fname, is_drawing, mouse_x, mouse_y

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        masks.append(masks[-1].copy())
        cv2.circle(masks[-1], (x, y), radius, 255, -1)
        img = np.where(np.expand_dims(masks[-1] == 255, -1), img_red, src)
        cv2.imshow(fname, img)

    if event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        img_cursor = img.copy()
        cv2.circle(img_cursor, (x, y), radius, (128, 128, 128))
        cv2.imshow(fname, img_cursor)

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
        if is_drawing:
            cv2.circle(masks[-1], (x, y), radius, 255, -1)
            img = np.where(np.expand_dims(masks[-1] == 255, -1), img_red, src)
            cv2.imshow(fname, img)
        else:
            img_cursor = img.copy()
            cv2.circle(img_cursor, (x, y), radius, (128, 128, 128))
            cv2.imshow(fname, img_cursor)


def plot_histogram(img, img_mask):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_rgb = [cv2.calcHist([img], [0], img_mask, [256], [0, 256]),
                cv2.calcHist([img], [1], img_mask, [256], [0, 256]),
                cv2.calcHist([img], [2], img_mask, [256], [0, 256])]
    hist_hsv = [cv2.calcHist([img_hsv], [0], img_mask, [180], [0, 180]),
                cv2.calcHist([img_hsv], [1], img_mask, [256], [0, 256]),
                cv2.calcHist([img_hsv], [2], img_mask, [256], [0, 256])]
    fig, axes = plt.subplots(nrows = 2, ncols = 2, sharex = False)
    axes[0, 0].plot(hist_hsv[0], label = 'H')
    axes[0, 0].legend()
    axes[0, 1].plot(hist_hsv[1], label = 'S')
    axes[0, 1].legend()
    axes[1, 0].plot(hist_hsv[2], label = 'V')
    axes[1, 0].legend()
    axes[1, 1].plot(hist_rgb[2], color = 'r', label = 'R')
    axes[1, 1].plot(hist_rgb[1], color = 'g', label = 'G')
    axes[1, 1].plot(hist_rgb[0], color = 'b', label = 'B')
    axes[1, 1].legend()
    plt.show()


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print("usage: python3 color_hist.py image_file")
        sys.exit()
    fname = args[1]
    src = cv2.imread(fname, 1)
    if src is None:
        print("error: cannot open file:", fname)
        sys.exit()
            
    points = []
    img = src.copy()
    img_red = np.zeros_like(src)
    img_red[:, :, 2] = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    masks = [np.zeros(src.shape[:2], dtype = np.uint8)]
    is_drawing = False
    radius = 16
    mouse_x, mouse_y = 0, 0

    # 取り込んだ画像の幅と高さを取得
    height, width = src.shape[:2]
    while height > 1000 or width > 1000:
        height = height * 0.9
        width = width * 0.9
    cv2.namedWindow(fname, cv2.WINDOW_NORMAL)  # ウィンドウをリサイズ可能に設定
    cv2.resizeWindow(fname, int(width), int(height))         # ウィンドウの初期サイズを設定
    
    cv2.imshow(fname, src)
    cv2.setMouseCallback(fname, mouse_event)

    print("[q] Quit, [Up] Thicken, [Down] Thin, [BackSpace] Undo, [Space] Histogram")
    while (True):
        key = cv2.waitKeyEx(1)
        if key == 113:                           # 'q'
            break
        elif not is_drawing:
            if key == 63232 and radius < 128:    # UP
                radius *= 2
                img_cursor = img.copy()
                cv2.circle(img_cursor, (mouse_x, mouse_y), radius, (128, 128, 128))
                cv2.imshow(fname, img_cursor)
            elif key == 63233 and radius > 2:    # DOWN
                radius //= 2
                img_cursor = img.copy()
                cv2.circle(img_cursor, (mouse_x, mouse_y), radius, (128, 128, 128))
                cv2.imshow(fname, img_cursor)
            elif key == 127 and len(masks) >= 2: # BackSapce
                masks.pop()
                img = np.where(np.expand_dims(masks[-1] == 255, -1), img_red, src)
                cv2.imshow(fname, img)
            elif key == 32 and len(masks) >=2:   # Space
                plot_histogram(src, masks[-1])
                del masks[1:]
                img = src.copy()
                cv2.imshow(fname, img)
##EOF
