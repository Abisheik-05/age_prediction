import { useRef, useState } from "react";

const ACCEPTED_TYPES = ["image/jpeg", "image/png"];
const MAX_FILE_SIZE = 5 * 1024 * 1024;

function UploadImage({ image, onImageSelected, disabled }) {
  const inputRef = useRef(null);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const validateAndSelect = (file) => {
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setError("Please choose a JPG, JPEG, or PNG image.");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError("Image size must be 5MB or smaller.");
      return;
    }
    setError("");
    onImageSelected(file);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    if (!disabled) validateAndSelect(event.dataTransfer.files[0]);
  };

  return (
    <div className="upload-section">
      <div
        className={`drop-zone ${isDragging ? "dragging" : ""} ${disabled ? "disabled" : ""}`}
        onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          className="file-input"
          type="file"
          accept="image/jpeg,image/png"
          onChange={(event) => validateAndSelect(event.target.files[0])}
          disabled={disabled}
        />
        {image ? (
          <div className="image-preview-wrap">
            <img src={image.previewUrl} alt="Selected face preview" className="preview" />
            <p className="image-name" title={image.file.name}>{image.file.name}</p>
          </div>
        ) : (
          <>
            <span className="upload-icon" aria-hidden="true">↥</span>
            <p>Drag and drop your photo here</p>
            <span>JPG, JPEG, or PNG · Maximum 5MB</span>
          </>
        )}
        <button className="secondary-btn" type="button" onClick={() => inputRef.current?.click()} disabled={disabled}>
          {image ? "Choose another image" : "Choose image"}
        </button>
      </div>
      {error && <p className="error-message" role="alert">{error}</p>}
    </div>
  );
}

export default UploadImage;
