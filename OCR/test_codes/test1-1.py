import easyocr
from PIL import Image, ImageDraw

# リーダーオブジェクトに日本語と英語を設定
reader = easyocr.Reader(['ja','en'],gpu = False)

# 画像のテキスト領域を解析
def analyze_picture(target_path: str):
    draw_chararea(target_path, reader.readtext(target_path))

# 入力画像内に文字列の領域を赤枠で囲う
def draw_chararea(target_path, results):
    image = Image.open(target_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    # 座標情報からテキスト領域を四角で囲う
    for result in results:
        print(result)
        p0, p1, p2, p3 = result[0]
        draw.line([ *p0, *p1, *p2, *p3, *p0], fill='red', width=3)
    image.save("draw_chararea_1.png")


if __name__ == "__main__":
    target_path = r".\samples\DSC_1936.JPG"
    print("test1-1")
    analyze_picture(target_path)