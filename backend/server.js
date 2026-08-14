const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs");
const path = require("path");

const app = express();

app.use(cors());
app.use(express.json());

const uploadDirectory = path.join(__dirname, "uploads");
fs.mkdirSync(uploadDirectory, { recursive: true });

// Multer Storage Configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDirectory);
  },

  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (["image/jpeg", "image/png"].includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Only JPG, JPEG, and PNG images are allowed."));
    }
  }
});

// Test Route
app.get("/", (req, res) => {
  res.send("Backend Running");
});

// Prediction Route
app.post("/api/predict", upload.single("image"), (req, res) => {

  console.log("================================");
  console.log("Image Received:");
  console.log(req.file);
  console.log("Saved File Path:", req.file?.path);
  console.log("================================");

  if (!req.file) {
    return res.status(400).json({
      message: "No image uploaded"
    });
  }

  // Future AI model integration goes here.
  // Pass req.file.path to a TensorFlow/PyTorch inference service and return its prediction.
  res.json({
    age: 25,
    confidence: 92,
    image: req.file.filename
  });
});

app.listen(5000, () => {
  console.log("Server running on port 5000");
});

app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError && error.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({ message: "Image size must be 5MB or smaller." });
  }
  if (error) return res.status(400).json({ message: error.message || "Unable to upload image." });
  next();
});
