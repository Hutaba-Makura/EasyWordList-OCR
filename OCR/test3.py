import cv2
import numpy as np
import test2 as T2
from PIL import Image

def correct_warp(image_path):
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        print("エラー: 画像が見つかりません。パスを確認してください。")
        return None
    
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_min, h_max = 42 , 97
    s_min, s_max = 1 , 30
    for y in range(img_hsv.shape[0]):
        for x in range(img_hsv.shape[1]):
            if not (img_hsv[y, x, 0] >= h_min and
                img_hsv[y, x, 0] <= h_max and
                img_hsv[y, x, 1] >= s_min and
                img_hsv[y, x, 1] <= s_max):
                img_hsv[y, x] = [0, 0, 0] # 黒にする
    # 画像を表示
    cv2.imshow("image", img_hsv)
    """
    # コントラストを調整
    #enhanced = cv2.equalizeHist(inverted)
    enhanced = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # enhancedを表示
    pil_image = Image.fromarray(enhanced)
    pil_image.show()
    """
    
    # エッジ検出
    edges = cv2.Canny(enhanced, 30, 150)
    
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
        cv2.imwrite(T2.reprint_filename(image_path), warped)
        return "corrected_image.png"
    else:
        print("ページの輪郭を検出できませんでした。")
        return None

# 使用例
if __name__ == "__main__":
    corrected_image = correct_warp(r".\samples\DSC_1931.JPG")
    if corrected_image:
        print("歪み補正後の画像を保存しました。")
