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
    output_path = 'image_without_exif.jpg'
    exif_data = remove_exif(image_path, output_path)
    with open(output_path, 'rb') as f:
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
    return response, output_path, exif_data

# 画像からEXIFメタデータを削除して保存
def remove_exif(image_path, output_path):
    """
    Returns:
        dict: A dictionary containing the original EXIF metadata.
    """
    image = Image.open(image_path)
    exif_data = {}

    # Extract EXIF metadata
    if image._getexif() is not None:
        exif_data = {
            ExifTags.TAGS.get(tag, tag): value
            for tag, value in image._getexif().items()
            if tag in ExifTags.TAGS
        }

    # Remove EXIF metadata
    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)
    image_without_exif.save(output_path)

    return exif_data

# OCRレスポンスからテキストのバウンディングボックスを取得
def extract_coords(response):
    annotations = response.text_annotations

    if not annotations:
        print("No text detected in the image.")
        return

    # 最初の要素は全文に関する情報なのでスキップ
    coords_list = []
    for annotation in annotations[1:]:
        # バウンディングボックスの頂点を取得
        vertices = annotation.bounding_poly.vertices

        # 頂点の座標をリストに変換　足りない場合は補完
        coords = []
        for vertex in vertices:
            coords.append([vertex.x, vertex.y])
        if len(coords) < 4:
            coords = np.pad(coords, ((0, 4 - len(coords)), (0, 0)), mode='constant')
        coords_list.append(coords)
    
    return coords_list

# 受け取った座標群から各領域のバウンディングボックスを赤く囲む
def draw_bounding_box(coords_list, image_path):
    image = cv2.imread(image_path)
    for points in coords_list:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imwrite('text_area_corrected.jpg', image)
    print("text_area_corrected.jpgを保存しました。")

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
            # NumPy配列に変換して2次元配列を保証
            coords_i = np.array(coords_list[i], dtype=np.float32)
            coords_j = np.array(coords_list[j], dtype=np.float32)
            distances = cdist(coords_i, coords_j)  # 領域iとjのすべての頂点間の距離を計算
            min_distance = np.min(distances)  # 最短距離を取得
            distance_matrix[i, j] = min_distance
            distance_matrix[j, i] = min_distance  # 対称行列

    return distance_matrix

# exif_dataをもとに画像と座標を回転させる
def rotate_image_and_coords(image_path, coords_list, exif_data):
    # 画像を読み込む
    image = cv2.imread(image_path)

    # 画像の向きを取得
    orientation = exif_data.get('Orientation', 1)

    # 画像の向きに応じて回転
    if orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 座標を回転
    if orientation == 3:
        for coords in coords_list:
            for point in coords:
                point[0] = image.shape[1] - point[0]
                point[1] = image.shape[0] - point[1]
    elif orientation == 6:
        for coords in coords_list:
            for point in coords:
                point[0], point[1] = point[1], image.shape[1] - point[0]
    elif orientation == 8:
        for coords in coords_list:
            for point in coords:
                point[0], point[1] = image.shape[0] - point[1], point[0]

    return image, coords_list

# クラスタリングして各領域をグループ化、各グループを青枠で囲う
# クラスタリング(領域間の最短距離が縦に25px以内、または横に50px以内の領域を同一クラスタとする)
def clustering(coords_list, image, threshold=100):
    distance_matrix = calculate_min_distance_matrix(coords_list)
    linkage_matrix = linkage(pdist(distance_matrix), method='ward') # 階層型クラスタリング
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance') # クラスタリング結果を取得
    # クラスタリング結果をもとに各領域を青枠で囲う
    for i, cluster in enumerate(clusters):
        color = (0, 255, 0)
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
response, output_path, exif_data = detect_text(image_path)
coords_list = extract_coords(response)
draw_bounding_box(coords_list, output_path)
    