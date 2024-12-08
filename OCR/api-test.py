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


# 結果をもとに各領域を赤枠で囲う
def draw_text_area(image_path, response):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]

    for text in response.text_annotations:
        vertices = text.bounding_poly.vertices
        # 座標変換：Cloud Vision (右上原点) -> OpenCV (左上原点)
        points = [
            (int(original_width-vertex.y), int(vertex.x))  # Y 座標を反転
            for vertex in vertices if vertex.x is not None and vertex.y is not None
        ]

        # 頂点の順序を修正して回転を正す←いらなくね？
        if len(points) == 4:
            cv2.polylines(image, [np.array(points, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2) # 赤枠
        else:
            print("頂点数が不足しているためスキップしました:", points)
    
    # 原点に黒丸を描画　50pxくらいの大きさ
    # cv2.circle(image, (0, 0), 50, (0, 0, 0), thickness=-1)

    # OCR結果から座標のみを抽出し、クラスタリングの準備
    coords = [
        np.mean(
            np.array([[original_width-vertex.y, vertex.x] for vertex in text.bounding_poly.vertices if vertex.x is not None and vertex.y is not None]),
            axis=0
        )
        for text in response.text_annotations
    ]

    #　文字が縦書きか横書きか判断
    # Bounding boxのx方向の長さの合計を計算
    x_length = 0
    for text in response.text_annotations:
        vertices = text.bounding_poly.vertices
        points = [
            (int(original_width-vertex.y), int(vertex.x))  # Y 座標を反転
            for vertex in vertices if vertex.x is not None and vertex.y is not None
        ]
        x_length += abs(points[0][0] - points[1][0])
    for text in response.text_annotations:
        vertices = text.bounding_poly.vertices
        points = [
            (int(original_width-vertex.y), int(vertex.x))  # Y 座標を反転
            for vertex in vertices if vertex.x is not None and vertex.y is not None
        ]
        y_length = abs(points[0][1] - points[1][1])
    is_vertical = x_length < y_length

    # 距離行列の作成とクラスタリング
    if not is_vertical:
        scaled_coords = [(coord[0] / 150, coord[1] / 75) for coord in coords]  # 横を100px単位、縦を50px単位で正規化
    else:
        scaled_coords = [(coord[0] / 75, coord[1] / 150) for coord in coords]
    distance_matrix = pdist(scaled_coords)  # 正規化された座標で距離を計算
    Z = linkage(distance_matrix, method='single')
    clusters = fcluster(Z, t=1, criterion='distance')  # 正規化した空間で閾値1を使用

    # 各クラスタに基づいて枠を描画
    for cluster_id in set(clusters):
        cluster_points = [coords[i] for i in range(len(coords)) if clusters[i] == cluster_id]
        # 各グループのバウンディングボックスを計算
        all_x = [int(p[0]) for p in cluster_points]
        all_y = [int(p[1]) for p in cluster_points]
        # 各x,yの最大最小の差が10px以内ならその差を30px分広げる
        if max(all_x) - min(all_x) < 15:
            all_x = [min(all_x) - 15, max(all_x) + 15]
        if max(all_y) - min(all_y) < 15:
            all_y = [min(all_y) - 15, max(all_y) + 15]
        bounding_box = [(min(all_x), min(all_y)), (max(all_x), max(all_y))]
        cv2.rectangle(image, bounding_box[0], bounding_box[1], (255, 0, 0), 3)  # 青枠


    cv2.imwrite('text_area.jpg', image)
    print("テキスト領域をtext_area.jpgに保存しました。")




# テキスト検出
image_path = r".\samples\DSC_1939.JPG"
result = detect_text(image_path)
draw_text_area(image_path, result)
