import os
from google.cloud import vision
from google.oauth2 import service_account
import cv2
import json
import numpy as np
from google.protobuf.json_format import MessageToDict

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


# 結果をもとに各領域を赤枠で囲う
def draw_text_area(image_path, response):
    image = cv2.imread(image_path)
    for text in response.text_annotations:
        vertices = text.bounding_poly.vertices
        points = [(vertex.y, vertex.x) for vertex in vertices if vertex.x is not None and vertex.y is not None]
        if len(points) == 4:
            cv2.polylines(image, [np.array(points, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
        else:
            print("頂点数が不足しているためスキップしました:", points)
    cv2.imwrite('text_area.jpg', image)
    print("テキスト領域をtext_area.jpgに保存しました。")

# テキスト検出
image_path = r".\samples\DSC_1937.JPG"
#result = detect_text(image_path)
#OCR_response.jsonに保存された結果を読み込む
with open('OCR_response.json', 'r', encoding='utf-8') as f:
    result = json.load(f)
draw_text_area(image_path, result)
