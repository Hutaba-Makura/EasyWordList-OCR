import cv2
import numpy as np
import os


def color_extract(image):
    # マスクを作成
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_min, h_max = 0, 115
    s_min, s_max = 2, 41
    
    img_mask = cv2.inRange(img_hsv, (h_min, s_min, 0), (h_max, s_max, 255))
    
    kernel = np.ones((5, 5), np.uint8)
    img_close = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)
    img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel)

    # 画像をコピーし、マスクの黒い部分を黒で塗りつぶす。残りは白になる
    img_masked = image.copy()
    img_masked[img_open == 0] = [0, 0, 0]
    #img_masked[img_open != 0] = [255, 255, 255]

    return img_masked

def contrast_enhance(image): # 画像のコントラストを強調する
    # カラー画像のコントラストを強調
    # 各チャンネルごとにヒストグラム均等化を行う
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    enhanced_img = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR) # YUVからBGRに変換
    return enhanced_img

def detect_edges(image):
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ぼかし処理でノイズ軽減
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Cannyエッジ検出
    edges = cv2.Canny(blurred, 50, 150)

    # Hough変換による直線検出
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # 直線を描画
    line_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 輪郭を近似するためのポイントを集約
    points = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])

    # 直線の交点を求める
    if len(points) > 0:
        points = np.array(points, dtype=np.float32)
        rect = cv2.minAreaRect(points)  # 最小外接矩形を求める
        box = cv2.boxPoints(rect)      # 矩形の4点を取得
        box = np.int32(box)

        # 結果を描画
        output_image = image.copy()
        cv2.drawContours(output_image, [box], 0, (0, 0, 255), 3)  # 赤で描画
        return output_image
    else:
        print("直線を検出できませんでした。")
        return image

def reprint_filename():
    global image_path
    target_path = image_path
    i = 1
    # pathを拡張子で分割
    root, ext = os.path.splitext(target_path)
    is_file = True
    while is_file:
        is_file = os.path.isfile(root + f"_paint_{i}" + ext)
        if(not is_file):
            break
        i += 1
    print(root + f"_paint_{i}" + ext)
    return root + f"_paint_{i}" + ext


def main():
    global image_path
    image_path = r"..\samples\DSC_1937.JPG"
    image = cv2.imread(image_path)
    #enhanced_img = contrast_enhance(image) # 却下
    img_bgr = color_extract(image)
    corrected_img = detect_edges(img_bgr)

    height, width = image.shape[:2]
    while height > 1000 or width > 1000:
        height = height * 0.9
        width = width * 0.9
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)           # ウィンドウをリサイズ可能に設定
    cv2.resizeWindow("test", int(width), int(height))    # ウィンドウの初期サイズを設定
    cv2.imshow("test", corrected_img)
    #cv2.imshow("test", img_bgr)
    #cv2.imshow("test", enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()