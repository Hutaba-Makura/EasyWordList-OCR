import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw

# リーダーオブジェクトに日本語と英語を設定
reader = easyocr.Reader(['ja','en'],gpu = True)

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

    return img_masked

def detect_edges(image, original_image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # グレースケール変換

    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # ぼかし処理でノイズ軽減

    edges = cv2.Canny(blurred, 50, 150) # Cannyエッジ検出

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
        box = cv2.boxPoints(rect)       # 矩形の4点を取得
        box = np.int32(box)

        # マスクを作成
        mask = np.zeros((original_image.shape[:2]), dtype=np.uint8)  # 1チャンネルのマスク
        cv2.drawContours(mask, [box], 0, 255, -1)  # マスクを白 (255) で塗りつぶす

        # マスクを適用
        corrected_img = cv2.bitwise_and(original_image, original_image, mask=mask)

        # 結果を描画
        output_image = corrected_img.copy()
        cv2.drawContours(output_image, [box], 0, (255, 0, 0), 3)  # 青で描画
        return output_image
    else:
        print("直線を検出できませんでした。")
        return original_image

# 入力画像内に文字列の領域を赤枠で囲う
def draw_chararea(img_path, results):
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)
    # 座標情報からテキスト領域を四角で囲う
    for result in results:
        # 文字情報のみを改行なしで表示
        print(result[1], end=" ")
        p0, p1, p2, p3 = result[0]
        draw.line([ *p0, *p1, *p2, *p3, *p0], fill='red', width=3)
    image.save("draw_chararea_5.png")


def main():
    global image_path
    image_path = r"..\samples\DSC_1937.JPG"
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    while height > 1000 or width > 1000:
        height = height * 0.9
        width = width * 0.9
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)           # ウィンドウをリサイズ可能に設定
    cv2.resizeWindow("test", int(width), int(height))    # ウィンドウの初期サイズを設定
    
    img_bgr = color_extract(image)
    corrected_img = detect_edges(img_bgr, image)
    # corrected_imgを保存
    cv2.imwrite("corrected_img.png", corrected_img)

    draw_chararea("corrected_img.png", reader.readtext("corrected_img.png"))

    img = cv2.imread("draw_chararea_5.png")
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()