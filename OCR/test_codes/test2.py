import easyocr
from PIL import Image, ImageDraw
import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import os

# description: 画像の文字認識を行うプログラムのサンプル

# リーダーオブジェクトに日本語と英語を設定
reader = easyocr.Reader(['ja', 'en'], gpu=True)


def analyze_picture(target_path: str):
    draw_chararea(target_path, reader.readtext(target_path, link_threshold=0.3,mag_ratio=1.1))

def reprint_filename(target_path: str):
    i = 1
    # pathを拡張子で分割
    root, ext = os.path.splitext(target_path)
    is_file = True
    while is_file:
        is_file = os.path.isfile(root + f"_{i}" + ext)
        if(not is_file):
            break
        i += 1
    return root + f"_paint_{i}" + ext

# 画像のテキスト領域を解析
def analyze_picture_tial(target_path: str):
    image = cv2.imread(target_path)
    # グレースケール化
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二値化（閾値は自動判別）
    ret, img_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    # 自動判別した閾値を表示
    print(f'閾値：{ret}')
    cv2.imwrite("thresh.png", img_thresh)
    draw_chararea(reprint_filename("thresh.png"), reader.readtext("thresh.png"))


# <追加>文字列の領域を赤枠で囲い、近いものはグループ化
def draw_chararea(target_path, results):
    image = Image.open(target_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    # テキストのみを取り出し、出力
    """
    for result in results:
        print(result[1])
    """
    
    # OCR結果から座標のみを抽出し、クラスタリングの準備
    coords = [np.mean(np.array(result[0]), axis=0) for result in results]  # 中心座標を使用
    
    # 距離行列の作成とクラスタリング
    distance_matrix = pdist(coords)
    Z = linkage(distance_matrix, method='single')
    clusters = fcluster(Z, t=50, criterion='distance')  # 50ピクセル以内を1グループに
    
    # 各クラスタに基づいて枠を描画
    for cluster_id in set(clusters):
        cluster_points = [results[i][0] for i in range(len(results)) if clusters[i] == cluster_id]
        # 各グループのバウンディングボックスを計算
        all_x = [p[0] for box in cluster_points for p in box]
        all_y = [p[1] for box in cluster_points for p in box]
        bounding_box = [(min(all_x), min(all_y)), (max(all_x), max(all_y))]
        draw.rectangle(bounding_box, outline="blue", width=3)
    
    # 単独の領域も赤枠で描画
    for result in results:
        p0, p1, p2, p3 = result[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill='red', width=1)

    image.save(reprint_filename("draw_chararea.png"))


if __name__ == "__main__":
    target_path = "test2.jpg"
    print("test")
    analyze_picture(target_path)
