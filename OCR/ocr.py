import os
from google.cloud import vision
from google.oauth2 import service_account
import cv2
import json
import numpy as np
from google.protobuf.json_format import MessageToDict
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from PIL import Image, ExifTags

# 環境変数からAPIキーのパスを取得
api_key_path = os.getenv('CloudVisionAPIKey')

if not api_key_path:
    raise ValueError("Environment variable 'CloudVisionAPIKey' is not set or empty.")

# 認証情報を読み込む
credentials = service_account.Credentials.from_service_account_file(api_key_path)

# クライアントのインスタンスを作成
client = vision.ImageAnnotatorClient(credentials=credentials)


# テキスト検出
def detect_text(image_path):
    with open(image_path, 'rb') as f:
        image = f.read()

    # Vision APIに渡す形式に変換
    image = vision.Image(content=image)

    # テキスト検出
    response = client.text_detection(image=image, image_context={"language_hints": ["ja", "en"]})

    # レスポンスを辞書形式に変換して保存
    response_dict = MessageToDict(response._pb, preserving_proto_field_name=True)
    with open('OCR_response.json', 'w', encoding='utf-8') as f:
        json.dump(response_dict, f, ensure_ascii=False, indent=4)

    print("レスポンスをOCR_response.jsonに保存しました。")
    return response

# Exif情報を元に画像を回転して補正
def correct_orientation(image_path):
    image = Image.open(image_path)
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    exif = image._getexif()
    if exif is not None and orientation in exif:
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    return image

# cloudVisionAPIから受け取った座標のスケールを元の画像のスケールに補正
def scale_coordinates(vertices, original_width, original_height, resized_width, resized_height):
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    return [
        (int(vertex.x * scale_x), int(vertex.y * scale_y)) for vertex in vertices
    ]

# 結果をもとに各領域を赤枠で囲う
def process_image(image_path, response):
    # 元画像のサイズを取得
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size
    
    # 回転補正を適用
    corrected_image = correct_orientation(image_path)
    corrected_image.save("corrected_image.jpg")  # 補正後の画像を保存
    corrected_width, corrected_height = corrected_image.size

    # OpenCVで補正後画像を読み込み
    image = cv2.imread("corrected_image.jpg")

    for text in response.text_annotations:
        vertices = text.bounding_poly.vertices
        # 座標変換とスケール補正
        points = scale_coordinates(vertices, corrected_width, corrected_height, original_width, original_height)
        points = [(int(x), int(y)) for x, y in points]  # OpenCV形式に変換
        # 頂点の順序を修正して回転を正す
        cv2.polylines(image, [np.array(points, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.imwrite('text_area_corrected.jpg', image)

    print("テキスト領域を囲んだ画像を保存しました。")
    
    return points, image

# 各領域間の最短距離を計算する距離行列を作成
def calculate_min_distance_matrix(coords_list):
    """
    領域間の最短距離を計算する距離行列を作成
    :param coords_list: 各領域のバウンディングボックスの頂点リスト
    :return: 距離行列（各領域間の最短距離）
    """
    num_coords = len(coords_list)
    distance_matrix = np.zeros((num_coords, num_coords))
    
    for i in range(num_coords):
        for j in range(i + 1, num_coords):
            distances = cdist(coords_list[i], coords_list[j])  # 領域iとjのすべての頂点間の距離を計算
            min_distance = np.min(distances)  # 最短距離を取得
            distance_matrix[i, j] = min_distance
            distance_matrix[j, i] = min_distance  # 対称行列

    return distance_matrix

# クラスタリングして各領域をグループ化、各グループを青枠で囲う
# クラスタリング(領域間の最短距離が縦に25px以内、または横に50px以内の領域を同一クラスタとする)
def clustering(coords_list, threshold=100):
    distance_matrix = calculate_min_distance_matrix(coords_list)
    linkage_matrix = linkage(pdist(distance_matrix), method='ward')
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
    # クラスタリング結果をもとに各領域を青枠で囲う
    image = cv2.imread("text_area_corrected.jpg")
    for i, cluster in enumerate(clusters):
        if cluster == 1:
            color = (0, 255, 0)  # 緑枠
        else:
            color = (0, 0, 255)  # 青枠
        points = coords_list[i]
        cv2.polylines(image, [np.array(points, dtype=np.int32)], isClosed=True, color=color, thickness=2)
    # クラスタリング結果を保存
    cv2.imwrite('text_area_clustered.jpg', image)
    print("クラスタリング結果を保存しました。")
    # 画像を表示
    cv2.imshow('image', image)
    return clusters

image_path = r".\samples\DSC_1937.JPG"

# メイン処理
response = detect_text(image_path)
coords_list, image = process_image(image_path, response)
clusters = clustering(coords_list)
cv2.imwrite('text_area_clustered.jpg', image)
print("クラスタリング結果を保存しました。")

    