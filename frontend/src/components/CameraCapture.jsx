import { useEffect, useRef, useState } from "react";

function CameraCapture({ onCapture, onRetake, disabled }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const previewUrlRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [capturedFile, setCapturedFile] = useState(null);
  const [error, setError] = useState("");

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCameraOn(false);
  };

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  const startCamera = async () => {
    setError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Your browser does not support camera access.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      if (!videoRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        throw new Error("Camera preview could not be initialized.");
      }
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setIsCameraOn(true);
    } catch (cameraError) {
      console.error("Camera Error:", cameraError);
      setError("Camera access was unavailable. Please allow access and try again.");
    }
  };

  const captureFace = () => {
    const video = videoRef.current;
    if (!video?.videoWidth) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (!blob) {
        setError("Unable to capture the current camera frame.");
        return;
      }

      const file = new File([blob], "captured-face.jpg", { type: "image/jpeg" });
      const previewUrl = URL.createObjectURL(blob);
      previewUrlRef.current = previewUrl;
      setCapturedImage(previewUrl);
      setCapturedFile(file);
      stopCamera();
      onCapture(file);
    }, "image/jpeg", 0.92);
  };

  const retakePhoto = () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setCapturedImage(null);
    setCapturedFile(null);
    onRetake();
    startCamera();
  };

  const showPreview = isCameraOn || capturedImage;

  return (
    <div className="camera-section">
      <div className={`camera-frame ${showPreview ? "" : "camera-frame-idle"}`}>
        <video
          ref={videoRef}
          className={isCameraOn ? "camera-video" : "camera-video hidden"}
          autoPlay
          playsInline
          muted
        />
        {capturedImage && <img src={capturedImage} alt="Captured face preview" className="camera-captured-image" />}
      </div>

      <div className="camera-actions">
        {!isCameraOn && !capturedImage && (
          <button className="secondary-btn" onClick={startCamera} disabled={disabled} type="button">Start Camera</button>
        )}
        {isCameraOn && (
          <button className="capture-btn" onClick={captureFace} disabled={disabled} type="button">Capture Face</button>
        )}
        {capturedFile && (
          <button className="secondary-btn" onClick={retakePhoto} disabled={disabled} type="button">Retake Photo</button>
        )}
      </div>
      {error && <p className="error-message" role="alert">{error}</p>}
    </div>
  );
}

export default CameraCapture;
