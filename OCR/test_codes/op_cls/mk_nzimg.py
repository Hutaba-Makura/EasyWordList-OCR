import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_binary_image(text, image_size=512, font_scale=5, font_thickness=10):
    """白い文字を黒背景に描画した二値画像を作成"""
    background = np.zeros((image_size, image_size), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (background.shape[1] - text_size[0]) // 2
    text_y = (background.shape[0] + text_size[1]) // 2
    cv2.putText(background, text, (text_x, text_y), font, font_scale, 255, font_thickness, lineType=cv2.LINE_AA)
    return background

def add_noise(image, noise_frequency):
    """ノイズを画像に追加"""
    noise = np.random.rand(*image.shape) < noise_frequency
    noisy_image = image.copy()
    noisy_image[noise] = 255 - noisy_image[noise]
    return noisy_image

def save_and_display_images(original_image, noisy_image, original_filename="original_ocr.png", noisy_filename="noisy_ocr.png"):
    """画像を保存し、表示"""
    cv2.imwrite(original_filename, original_image)
    cv2.imwrite(noisy_filename, noisy_image)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Bolder Binary Image")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Noisy Image with Moderate Frequency")
    plt.imshow(noisy_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Step 1: Create a bold binary image
    text = "OCR"
    font_thickness_bold = 20
    binary_image = create_binary_image(text, font_thickness=font_thickness_bold)

    # Step 2: Add moderate noise
    noise_frequency = 0.001  # 0.1% of the pixels will be noisy
    noisy_image = add_noise(binary_image, noise_frequency)

    # Step 3: Save and display the images
    save_and_display_images(binary_image, noisy_image)
