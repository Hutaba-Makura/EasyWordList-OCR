import easyocr
from PIL import Image, ImageDraw
import cv2

# リーダーオブジェクトに日本語と英語を設定
reader = easyocr.Reader(['ja','en'],gpu = True)

# 画像を処理してからOCRを実行
def analyze_picture_tial(target_path: str):
    image = cv2.imread(target_path)
    # グレースケール化
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二値化（閾値は自動判別）
    ret, img_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    # 自動判別した閾値を表示
    print(f'閾値：{ret}')
    cv2.imwrite("monochrome.png", img_thresh)
    draw_chararea("monochrome.png", reader.readtext("monochrome.png"))

# 入力画像内に文字列の領域を赤枠で囲う
def draw_chararea(target_path, results):
    image = Image.open(target_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    # 座標情報からテキスト領域を四角で囲う
    for result in results:
        # 文字情報のみを改行なしで表示
        print(result[1], end=" ")
        p0, p1, p2, p3 = result[0]
        draw.line([ *p0, *p1, *p2, *p3, *p0], fill='red', width=3)
    image.save("draw_chararea_2.png")


if __name__ == "__main__":
    target_path = r"..\samples\DSC_1936.JPG"
    print("test1-2")
    analyze_picture_tial(target_path)