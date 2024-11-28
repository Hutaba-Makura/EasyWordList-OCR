import cv2


# Description: 画像のコントラストを強調するプログラム
def test_image(image_path):
    print("画像を読み込んでいます...")
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        print("エラー: 画像が見つからません。パスを確認してください。")
        return

    # 画像をいい感じの大きさで表示
    height, width = image.shape[:2] # 取り込んだ画像の幅と高さを取得
    while height > 1000 or width > 1000:
        height = height * 0.9
        width = width * 0.9
    cv2.namedWindow(image_path, cv2.WINDOW_NORMAL)  # ウィンドウをリサイズ可能に設定
    cv2.resizeWindow(image_path, int(width), int(height))         # ウィンドウの初期サイズを設定

    # カラー画像のコントラストを強調
    # 各チャンネルごとにヒストグラム均等化を行う
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    enhanced = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    # enhancedを表示
    cv2.imshow(image_path, enhanced)

if __name__ == "__main__":
    test_image(r".\samples\DSC_1935.JPG")
    cv2.waitKey(0)