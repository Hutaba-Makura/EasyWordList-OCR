import easyocr
import os

# リーダーオブジェクトに日本語と英語を設定
reader = easyocr.Reader(['en', 'ja'])

# EasyOCRを使って画像から文字データを抽出し結果を出力
def analyze_picture(target_path_1: str):
    if not os.path.exists(target_path_1):
        print("Error: The specified image file does not exist.")
        exit(1)
    results = reader.readtext(target_path_1)
    for result in results:
        print(result)

# 試し実行用
if __name__ == "__main__":
    target_path_1 = r"C:\Users\src12\Documents\GitHub\EasyWordList-OCR\DSC_0854.JPG"
    analyze_picture(target_path_1)