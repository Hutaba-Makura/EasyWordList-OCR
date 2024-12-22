import json

# OCR_response.jsonに保存された結果を読み込む
def load_response():
    with open('OCR_response.json', 'r', encoding='utf-8') as f:
        response_dict = json.load(f)
    return response_dict

# バウンディングボックスの数を数える
def count_bounding_boxes(response_dict):
    # textAnnotations を辞書から取得
    text_annotations = response_dict.get('text_annotations', [])
    
    # 最初の要素を除いたリストの長さを返す
    return len(text_annotations[1:])

if __name__ == '__main__':
    response_dict = load_response()
    print("バウンディングボックスの数:", count_bounding_boxes(response_dict))
