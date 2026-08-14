import { useEffect, useRef, useState } from "react";
import "./App.css";
import CameraCapture from "./components/CameraCapture";
import Loader from "./components/Loader";
import ResultCard from "./components/ResultCard";
import UploadImage from "./components/UploadImage";
import { predictAge } from "./services/predictionService";

function App() {
  const [mode, setMode] = useState("upload");
  const [uploadImage, setUploadImage] = useState(null);
  const [capturedFile, setCapturedFile] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const uploadPreviewRef = useRef(null);

  useEffect(() => () => {
    if (uploadPreviewRef.current) URL.revokeObjectURL(uploadPreviewRef.current);
  }, []);

  const handleImageSelected = (file) => {
    if (uploadPreviewRef.current) URL.revokeObjectURL(uploadPreviewRef.current);
    const previewUrl = URL.createObjectURL(file);
    uploadPreviewRef.current = previewUrl;
    setUploadImage({ file, previewUrl });
    setPredictionResult(null);
    setError("");
  };

  const handleModeChange = (nextMode) => {
    if (uploadPreviewRef.current) URL.revokeObjectURL(uploadPreviewRef.current);
    uploadPreviewRef.current = null;
    setMode(nextMode);
    setUploadImage(null);
    setCapturedFile(null);
    setPredictionResult(null);
    setError("");
  };

  const handlePrediction = async () => {
    const imageFile = mode === "upload" ? uploadImage?.file : capturedFile;
    if (!imageFile) return;

    setIsLoading(true);
    setError("");
    setPredictionResult(null);

    try {
      const prediction = await predictAge(imageFile);
      setPredictionResult({ ...prediction, timestamp: new Date() });
    } catch (requestError) {
      setError(requestError.message || "Unable to analyze this image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="card" aria-labelledby="app-title">
        <div className="brand-mark" aria-hidden="true">AI</div>
        <h1 id="app-title" className="title">Age Prediction AI</h1>
        <p className="subtitle">Predict age from facial images using AI</p>

        <div className="mode-tabs" role="tablist" aria-label="Prediction input mode">
          <button
            className={`tab ${mode === "upload" ? "active" : ""}`}
            onClick={() => handleModeChange("upload")}
            type="button"
            role="tab"
            aria-selected={mode === "upload"}
          >
            Upload image
          </button>
          <button
            className={`tab ${mode === "camera" ? "active" : ""}`}
            onClick={() => handleModeChange("camera")}
            type="button"
            role="tab"
            aria-selected={mode === "camera"}
          >
            Live camera
          </button>
        </div>

        {mode === "upload" ? (
          <UploadImage image={uploadImage} onImageSelected={handleImageSelected} disabled={isLoading} />
        ) : (
          <CameraCapture
            onCapture={(file) => { setCapturedFile(file); setPredictionResult(null); setError(""); }}
            onRetake={() => { setCapturedFile(null); setPredictionResult(null); setError(""); }}
            disabled={isLoading}
          />
        )}

        {(mode === "upload" ? uploadImage : capturedFile) && (
          <button className="predict-btn" onClick={handlePrediction} disabled={isLoading} type="button">
            {isLoading ? "Analyzing..." : "Predict Age"}
          </button>
        )}

        {isLoading && <Loader />}
        {error && <p className="error-message" role="alert">{error}</p>}
        {predictionResult && <ResultCard result={predictionResult} />}
      </section>
    </main>
  );
}

export default App;
