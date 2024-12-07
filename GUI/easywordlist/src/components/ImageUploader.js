import React, { useState } from "react";
import axios from "axios";

const ImageUploader = ({ setProcessedData }) => {
    const [image, setImage] = useState(null);
    const [isUploading, setIsUploading] = useState(false); // ロード中の状態
    const [error, setError] = useState(""); // エラーメッセージ
    const [successMessage, setSuccessMessage] = useState(""); // 成功メッセージ

    // 画像選択時に呼ばれる関数
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setImage(file);
            setSuccessMessage(""); // 新たに画像を選択した場合、成功メッセージをリセット
        }
    };

    // 画像をアップロードする関数
    const handleUpload = async () => {
        if (!image) return;

        const formData = new FormData();
        formData.append("image", image);

        setIsUploading(true); // アップロード中にセット

        try {
            const response = await axios.post("http://localhost:5000/api/ocr", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            setProcessedData(response.data); // 画像と領域データを保存
            setSuccessMessage("画像のアップロードと処理が成功しました！");
            setError(""); // エラーがあればリセット
        } catch (error) {
            console.error("Error uploading image:", error);
            setError("画像のアップロードに失敗しました。再試行してください。");
            setSuccessMessage(""); // 成功メッセージをリセット
        } finally {
            setIsUploading(false); // アップロード完了後にリセット
        }
    };

    return (
        <div>
            <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                disabled={isUploading} // アップロード中は選択できないようにする
            />
            <button onClick={handleUpload} disabled={!image || isUploading}>
                {isUploading ? "アップロード中..." : "アップロード"}
            </button>

            {/* エラーメッセージの表示 */}
            {error && <div style={{ color: "red", marginTop: "10px" }}>{error}</div>}

            {/* 成功メッセージの表示 */}
            {successMessage && <div style={{ color: "green", marginTop: "10px" }}>{successMessage}</div>}
        </div>
    );
};

export default ImageUploader;

