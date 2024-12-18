// OCR結果画像の表示
import React from "react";

const ImageDisplay = ({ imageSrc, regions, onRegionClick }) => {
    return (
        <div style={{ position: "relative" }}>
            <img src={imageSrc} alt="OCR結果" style={{ maxWidth: "100%" }} />
            {regions.map((region, index) => (
                <div
                    key={index}
                    onClick={() => onRegionClick(region)}
                    style={{
                        position: "absolute",
                        top: region.top,
                        left: region.left,
                        width: region.width,
                        height: region.height,
                        border: "2px solid red",
                        cursor: "pointer",
                    }}
                />
            ))}
        </div>
    );
};

export default ImageDisplay;
