from deepface import DeepFace

result = DeepFace.analyze(
    img_path="backend/uploads/1786679359829-camera-capture-1786679357276.jpg",
    actions=["age"],
    detector_backend="retinaface",
    enforce_detection=False,
)

print(result)
