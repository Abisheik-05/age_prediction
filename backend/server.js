const express = require("express");
const axios = require("axios");
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
app.post("/api/predict", upload.single("image"), async (req, res) => {
  try {
    console.log("================================");
    console.log("Image Received:");
    console.log(req.file);
    console.log("Saved File Path:", req.file?.path);
    console.log("================================");

    if (!req.file) {
      return res.status(400).json({
        message: "No image uploaded",
      });
    }

    const flaskResponse = await axios.post(
      "http://127.0.0.1:5001/predict-age",
      {
        image_path: req.file.path,
      }
    );

    const predictedAge = flaskResponse.data.age;

    res.json({
      age: predictedAge,
      confidence: 100,
      image: req.file.filename,
    });
  } catch (error) {
    console.error("DeepFace Error:", error.message);

    res.status(500).json({
      message: "Age prediction failed",
      error: error.message,
    });
  }
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
