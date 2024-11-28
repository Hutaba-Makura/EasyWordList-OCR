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

    # 画像をコピーし、マスクの黒い部分を黒で塗りつぶす
    img_masked = image.copy()
    img_masked[img_open == 0] = [0, 0, 0]

    return img_masked

def contrast_enhance(image): # 画像のコントラストを強調する
    # カラー画像のコントラストを強調
    # 各チャンネルごとにヒストグラム均等化を行う
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    enhanced_img = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR) # YUVからBGRに変換
    return enhanced_img

def detect_edges(image): # 画像のエッジを検出する
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # エッジ検出
    edges = cv2.Canny(gray, 30, 150)
    
    # 輪郭検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 最大の四角形の輪郭を取得
    page_contour = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            page_contour = approx
            break

    # ページが見つかった場合のみ透視変換を実施
    if page_contour is not None:
        pts = page_contour.reshape(4, 2)
        
        # 左上、右上、右下、左下の順に並べる
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # 出力画像のサイズを決定
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # 変換後の画像の4点を定義
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # 透視変換マトリックスを計算し、画像を変換
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        # 結果を保存
        cv2.imwrite(reprint_filename(), warped)
        return warped
    else:
        print("ページの輪郭を検出できませんでした。")
        return None
    
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
    image_path = r".\samples\DSC_1938.JPG"
    image = cv2.imread(image_path)
    #enhanced_img = contrast_enhance(image) # 却下
    img_bgr = color_extract(image)
    #corrected_img = detect_edges(img_bgr)

    height, width = image.shape[:2]
    while height > 1000 or width > 1000:
        height = height * 0.9
        width = width * 0.9
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)           # ウィンドウをリサイズ可能に設定
    cv2.resizeWindow("test", int(width), int(height))    # ウィンドウの初期サイズを設定
    #cv2.imshow("corrected_img", corrected_img)
    cv2.imshow("test", img_bgr)
    #cv2.imshow("test", enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()