import os
from google.cloud import vision
from google.oauth2 import service_account

# 環境変数からAPIキーのパスを取得
api_key_path = os.getenv('CloudVisionAPIKey')
print(api_key_path)
"""

if not api_key_path:
    raise ValueError("Environment variable 'CloudVisionAPIKey' is not set or empty.")
"""
    
# 認証情報を読み込む
credentials = service_account.Credentials.from_service_account_file(r"C:\Users\syaro\OneDrive\デスクトップ\C105同人誌\API-Key\key.json")

# クライアントのインスタンスを作成
client = vision.ImageAnnotatorClient(credentials=credentials)

# 画像ファイルの読み込み
def detect_text(image_path):
    with open(image_path, 'rb') as f:
        image = f.read()

    # Vision APIに渡す形式に変換
    image = vision.Image(content=image)

    # テキスト検出
    response = client.text_detection(image=image)

    # 検出結果の取得
    texts = response.text_annotations

    # 検出されたテキストを表示
    for text in texts:
        print(text.description)

    return texts

# テキスト検出
image_path = r".\samples\DSC_1937.JPG"
detect_text(image_path)
