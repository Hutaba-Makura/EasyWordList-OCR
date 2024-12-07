// バックエンドとの通信処理
import axios from "axios";

const saveDataToBackend = async (data) => {
    try {
        await axios.post("/api/save", data);
        alert("データが保存されました！");
    } catch (error) {
        console.error("Error saving data:", error);
    }
};
