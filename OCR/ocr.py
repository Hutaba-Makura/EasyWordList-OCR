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
    return response, exif_data, cv2.imread(output_path)

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
def draw_bounding_box(coords_list, image, color):
    for points in coords_list:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], isClosed=True, color=color, thickness=2)
    print("バウンディングボックスを描画しました。")

# exif_dataをもとに画像と座標を回転させる
def rotate_image_and_coords(image, coords_list, exif_data):
    # 画像の向きを取得
    orientation = exif_data.get('Orientation', 1)
    print(f"Orientation: {orientation}")

    # 画像の向きに応じて回転
    if orientation == 3: # 180度回転
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 6: # 反時計回りに90度回転
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8: # 時計回りに90度回転
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # バウンディングボックスの座標を回転
    if orientation == 3:
        for coords in coords_list:
            for point in coords:
                point[0] = image.shape[1] - point[0]
                point[1] = image.shape[0] - point[1]
    elif orientation == 6:
        for coords in coords_list:
            for point in coords:
                point[0], point[1] = image.shape[1] - point[1], point[0]
    elif orientation == 8:
        for coords in coords_list:
            for point in coords:
                point[0], point[1] = point[1], image.shape[0] - point[0]
    
    return image, coords_list

def cluster_bounding_boxes(coords_list, threshold):
    """
    Clusters bounding boxes based on horizontal, vertical proximity thresholds, and minimum distance.

    Args:
        coords_list (list): List of bounding box coordinate arrays.
        width (int): Maximum horizontal distance to consider for clustering.
        height (int): Maximum vertical distance to consider for clustering.
        threshold (float): Minimum distance threshold for clustering.

    Returns:
        list: List of clustered bounding boxes, where each cluster is represented by a single bounding box.
    """
    # 各バウンディングボックスの中心を計算
    centroids = []
    for coords in coords_list:
        x_coords = [point[0] for point in coords]
        y_coords = [point[1] for point in coords]
        centroid = [np.mean(x_coords), np.mean(y_coords)]
        centroids.append(centroid)

    # 中心間の距離を計算
    pairwise_distances = cdist(centroids, centroids, metric='euclidean')

    # クラスタリング
    linkage_matrix = linkage(pdist(centroids), method='single')

    # クラスタリングの結果を取得
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')


    # クラスタごとにバウンディングボックスをマージ
    clustered_boxes = []
    for cluster_id in set(clusters):
        cluster_coords = [coords_list[i] for i in range(len(coords_list)) if clusters[i] == cluster_id]

        # バウンディングボックスをマージ
        all_x = [point[0] for box in cluster_coords for point in box]
        all_y = [point[1] for box in cluster_coords for point in box]
        merged_box = [
            [min(all_x), min(all_y)],
            [max(all_x), min(all_y)],
            [max(all_x), max(all_y)],
            [min(all_x), max(all_y)]
        ]
        clustered_boxes.append(merged_box)

    return clustered_boxes


def main():
    image_path = r".\samples\DSC_1937.JPG"

    # メイン処理
    response, exif_data, image = detect_text(image_path)
    coords_list = extract_coords(response)
    draw_bounding_box(coords_list, image, (0, 0, 255)) # 赤色で描画
    image, coords_list = rotate_image_and_coords(image, coords_list, exif_data) # 画像と座標を回転
    clustered_boxes = cluster_bounding_boxes(coords_list, threshold=30)
    draw_bounding_box(clustered_boxes, image, (255, 0, 0)) # 青色で描画
    clustered_boxes = cluster_bounding_boxes(clustered_boxes, threshold=30)
    draw_bounding_box(clustered_boxes, image, (0, 255, 0)) # 緑色で描画
    # 50ピクセルの長さの緑色の線を引く
    cv2.line(image, (0, 50), (50, 50), (0, 255, 0), 2)
    # 画像を保存
    cv2.imwrite('text_area_corrected.jpg', image)
    print("text_area_corrected.jpgを保存しました。")
    
if __name__ == '__main__':
    main()