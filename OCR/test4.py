import cv2
import numpy as np


def color_extract(image): # 画像中の本のページ以外を黒くする
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_min, h_max = 0 , 115
    s_min, s_max = 2 , 41
    for y in range(img_hsv.shape[0]):
        for x in range(img_hsv.shape[1]):
            if not (img_hsv[y, x, 0] >= h_min and
                img_hsv[y, x, 0] <= h_max and
                img_hsv[y, x, 1] >= s_min and
                img_hsv[y, x, 1] <= s_max):
                img_hsv[y, x] = [0, 0, 0] # 黒にする
    
    kernel = np.ones((5, 5), np.uint8) # カーネルの定義 5x5の正方行列
    img_close = cv2.morphologyEx(img_hsv, cv2.MORPH_CLOSE, kernel) # クロージング処理
    img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel) # オープニング処理

    # HSVからBGRに変換して表示
    img_bgr = cv2.cvtColor(img_open, cv2.COLOR_HSV2BGR)
    return img_bgr

def contrast_enhance(image): # 画像のコントラストを強調する
    # カラー画像のコントラストを強調
    # 各チャンネルごとにヒストグラム均等化を行う
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    enhanced = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    return enhanced