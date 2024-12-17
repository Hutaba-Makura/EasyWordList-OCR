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
from itertools import combinations

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
    Clusters bounding boxes based on proximity thresholds.

    Args:
        coords_list (list): List of bounding box coordinate arrays.
        threshold (float): Minimum distance threshold for clustering.

    Returns:
        list: List of clustered bounding boxes.
    """
    # 各バウンディングボックスの中心を計算
    centroids = []
    for coords in coords_list:
        # 入力が単純なバウンディングボックス（4頂点）であることを確認
        x_coords = [point[0] for point in coords]
        y_coords = [point[1] for point in coords]
        centroid = [np.mean(x_coords), np.mean(y_coords)]  # 中心座標を計算
        centroids.append(centroid)

    # クラスタリング
    linkage_matrix = linkage(pdist(centroids), method='single')
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

def calculate_box_distance(box1, box2):
    """
    2つのバウンディングボックスの最短距離を計算します。

    Returns:    
        float: 2つのバウンディングボックスの最短距離。
    """
    # Extract all edges (line segments) of each box
    def get_edges(box):
        return [(box[i], box[(i + 1) % 4]) for i in range(4)]

    edges1 = get_edges(box1)
    edges2 = get_edges(box2)
    
    # Compute the minimum distance between all edges
    min_distance = float('inf')
    for (p1, q1) in edges1:
        for (p2, q2) in edges2:
            distance = segment_distance(p1, q1, p2, q2)
            min_distance = min(min_distance, distance)
    return min_distance

def segment_distance(p1, q1, p2, q2):
    """
    2つの線分の最短距離を計算します。

    Returns:
        float: 2つの線分の最短距離。
    """
    def point_to_segment_distance(p, a, b):
        # Project point p onto line segment a-b
        ab = np.array(b) - np.array(a)
        ap = np.array(p) - np.array(a)
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))  # Clamp t to the segment
        closest = a + t * ab
        return np.linalg.norm(p - closest)
    
    # Distance between all combinations of endpoints
    distances = [
        point_to_segment_distance(np.array(p1), np.array(p2), np.array(q2)),
        point_to_segment_distance(np.array(q1), np.array(p2), np.array(q2)),
        point_to_segment_distance(np.array(p2), np.array(p1), np.array(q1)),
        point_to_segment_distance(np.array(q2), np.array(p1), np.array(q1))
    ]
    return min(distances)

def is_overlapping(box1, box2):
    """
    Check if two bounding boxes overlap.
    
    Args:
        box1, box2: List of coordinates [[x1, y1], ..., [x4, y4]].

    Returns:
        bool: True if the boxes overlap, otherwise False.
    """
    x1_min = min(p[0] for p in box1)
    x1_max = max(p[0] for p in box1)
    y1_min = min(p[1] for p in box1)
    y1_max = max(p[1] for p in box1)

    x2_min = min(p[0] for p in box2)
    x2_max = max(p[0] for p in box2)
    y2_min = min(p[1] for p in box2)
    y2_max = max(p[1] for p in box2)

    # Check for overlap
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def merge_boxes_with_overlap(boxes, threshold):
    """
    Merge bounding boxes considering both overlap and distance.

    Args:
        boxes (list): List of bounding boxes [[x1, y1], ..., [x4, y4]].
        threshold (float): Distance threshold for merging.

    Returns:
        list: List of merged bounding boxes.
    """
    merged = []
    while boxes:
        base_box = boxes.pop(0)
        i = 0
        while i < len(boxes):
            box = boxes[i]
            # Check overlap or distance
            if is_overlapping(base_box, box) or calculate_box_distance(base_box, box) < threshold:
                # Merge boxes by taking the outer bounds
                all_x = [p[0] for p in base_box + box]
                all_y = [p[1] for p in base_box + box]
                base_box = [
                    [min(all_x), min(all_y)],
                    [max(all_x), min(all_y)],
                    [max(all_x), max(all_y)],
                    [min(all_x), max(all_y)]
                ]
                boxes.pop(i)  # Remove the merged box
            else:
                i += 1
        merged.append(base_box)
    return merged

# API用のmain関数
def ocr_document(image_path):
     # APIを使ってテキストを検出, exif_dataも取得
    response, exif_data, image = detect_text(image_path)
    # テキストのバウンディングボックスの座標群を取得
    coords_list = extract_coords(response)
    draw_bounding_box(coords_list, image, (0, 0, 255)) # 赤色で描画
    # 画像と座標を回転
    image, coords_list = rotate_image_and_coords(image, coords_list, exif_data)

    # 一回目のクラスタリング
    clustered_boxes = cluster_bounding_boxes(coords_list, threshold=80) # 早いけど精度が低い
    # clustered_boxes = merge_boxes_with_overlap(coords_list, threshold=10) # 遅いけど精度が高い
    #draw_bounding_box(clustered_boxes, image, (255, 0, 0)) # 青色で描画

    # 二回目のクラスタリング
    merged_boxes = merge_boxes_with_overlap(clustered_boxes, threshold=10)
    draw_bounding_box(merged_boxes, image, (255, 0, 0)) # 青色で描画

    # 画像を保存
    cv2.imwrite('text_area_corrected.jpg', image)
    print("text_area_corrected.jpgを保存しました。")

    return coords_list, merged_boxes


def main():
    image_path = r".\samples\DSC_1936.JPG"

    # メイン処理
    response, exif_data, image = detect_text(image_path)
    coords_list = extract_coords(response)
    draw_bounding_box(coords_list, image, (0, 0, 255)) # 赤色で描画
    image, coords_list = rotate_image_and_coords(image, coords_list, exif_data) # 画像と座標を回転
    clustered_boxes = cluster_bounding_boxes(coords_list, threshold=80) # 早いけど精度が低い
    # clustered_boxes = merge_boxes_with_overlap(coords_list, threshold=10) # 遅いけど精度が高い
    #draw_bounding_box(clustered_boxes, image, (255, 0, 0)) # 青色で描画
    merged_boxes = merge_boxes_with_overlap(clustered_boxes, threshold=10)
    draw_bounding_box(merged_boxes, image, (255, 0, 0)) # 青色で描画
    # 50ピクセルの長さの緑色の線を引く
    cv2.line(image, (0, 50), (50, 50), (0, 255, 0), 2)
    # 画像を保存
    cv2.imwrite('text_area_corrected.jpg', image)
    print("text_area_corrected.jpgを保存しました。")
    
if __name__ == '__main__':
    main()