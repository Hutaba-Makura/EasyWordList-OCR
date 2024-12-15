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

def draw_bounding_boxes(image_path, response):
    # 画像を読み込む
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # OCRレスポンスからテキストのバウンディングボックスを取得
    annotations = response.text_annotations

    if not annotations:
        print("No text detected in the image.")
        return

    # 最初の要素は全文に関する情報なのでスキップ
    for annotation in annotations[1:]:
        # バウンディングボックスの頂点を取得
        vertices = annotation.bounding_poly.vertices

        # 頂点の座標をリストに変換
        pts = np.array([[vertex.x, vertex.y] for vertex in vertices], np.int32)

        # 頂点が不足している場合は補完
        if len(pts) < 4:
            pts = np.pad(pts, ((0, 4 - len(pts)), (0, 0)), mode='constant')

        # バウンディングボックスを描画
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 画像を保存または表示
    output_path = "output_with_boxes.jpg"
    cv2.imwrite(output_path, image)
    print(f"バウンディングボックスを描画した画像を保存しました: {output_path}")

# 実行例
if __name__ == "__main__":
    image_path = r".\tes.jpg"

    # OCRレスポンスを取得
    response = detect_text(image_path)

    # バウンディングボックスを描画
    draw_bounding_boxes(image_path, response)