// テキスト領域の選択
import React, { useState } from "react";

const RegionSelector = ({ selectedRegion, onSave }) => {
    const [word, setWord] = useState("");
    const [meaning, setMeaning] = useState("");

    const handleSave = () => {
        onSave({ word, meaning, ...selectedRegion });
        setWord("");
        setMeaning("");
    };

    return (
        <div>
            <h3>領域の単語と意味を入力</h3>
            <input
                type="text"
                placeholder="単語"
                value={word}
                onChange={(e) => setWord(e.target.value)}
            />
            <input
                type="text"
                placeholder="意味"
                value={meaning}
                onChange={(e) => setMeaning(e.target.value)}
            />
            <button onClick={handleSave} disabled={!word || !meaning}>
                保存
            </button>
        </div>
    );
};

export default RegionSelector;
