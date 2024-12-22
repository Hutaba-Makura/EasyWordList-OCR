import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "noisy_ocr.png"

def save_and_display_images(original, intermediate, final, titles, output_paths):
    """画像を保存し、表示"""
    plt.figure(figsize=(18, 6))
    for i, (image, title, path) in enumerate(zip([original, intermediate, final], titles, output_paths)):
        plt.subplot(1, 3, i + 1)
        plt.title(title)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        if path:  # 保存が必要な場合
            cv2.imwrite(path, image)
    plt.tight_layout()
    plt.show()

def process_morphology(image_path, kernel_size, operation_name, output_prefix):
    """オープニング・クロージングの処理を膨張と収縮を含めて保存・表示"""
    try:
        # 画像の読み込み
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        
        # カーネルの定義
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 収縮 (erosion) 処理
        eroded = cv2.erode(image, kernel)
        erosion_path = f"{output_prefix}_erosion.png"
        
        # 膨張 (dilation) 処理
        dilated = cv2.dilate(image, kernel)
        dilation_path = f"{output_prefix}_dilation.png"
        
        # モルフォロジー操作
        if operation_name == "opening":
            final_image = cv2.dilate(eroded, kernel)
        elif operation_name == "closing":
            final_image = cv2.erode(dilated, kernel)
        else:
            raise ValueError(f"Unsupported operation: {operation_name}")
        
        final_path = f"{output_prefix}_final.png"
        
        # 保存と表示
        save_and_display_images(
            original=image,
            intermediate=eroded if operation_name == "opening" else dilated,
            final=final_image,
            titles=[
                "Original Image",
                "Erosion (Opening Step)" if operation_name == "opening" else "Dilation (Closing Step)",
                "Final Image (Opening)" if operation_name == "opening" else "Final Image (Closing)"
            ],
            output_paths=[None, erosion_path if operation_name == "opening" else dilation_path, final_path]
        )
        print(f"Images saved:\n - Erosion/Dilation: {erosion_path if operation_name == 'opening' else dilation_path}\n - Final: {final_path}")
        
    except Exception as e:
        print(f"Error during {operation_name} processing: {e}")

def opening(image_path, kernel_size=5):
    """オープニング処理"""
    process_morphology(image_path, kernel_size, "opening", "opening")

def closing(image_path, kernel_size=5):
    """クロージング処理"""
    process_morphology(image_path, kernel_size, "closing", "closing")

if __name__ == "__main__":
    opening(image_path)
    closing(image_path)
