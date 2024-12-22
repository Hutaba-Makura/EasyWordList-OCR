import React, { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import ImageDisplay from "./components/ImageDisplay";
import RegionSelector from "./components/RegionSelector";

const App = () => {
    const [processedData, setProcessedData] = useState(null);
    const [selectedRegion, setSelectedRegion] = useState(null);
    const [savedData, setSavedData] = useState([]);

    const handleRegionClick = (region) => {
        setSelectedRegion(region);
    };

    const handleSave = (data) => {
        setSavedData([...savedData, data]);
    };

    return (
        <div>
            <h1>OCRアプリ</h1>
            <ImageUploader setProcessedData={setProcessedData} />
            {processedData && (
                <ImageDisplay
                    imageSrc={processedData.image}
                    regions={processedData.regions}
                    onRegionClick={handleRegionClick}
                />
            )}
            {selectedRegion && (
                <RegionSelector
                    selectedRegion={selectedRegion}
                    onSave={handleSave}
                />
            )}
            <button
                onClick={() => saveDataToBackend(savedData)}
                disabled={savedData.length === 0}
            >
                データ保存
            </button>
        </div>
    );
};

export default App;


