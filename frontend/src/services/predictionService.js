const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function predictAge(imageFile) {
  const formData = new FormData();
  formData.append("image", imageFile);

  const response = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const message = data?.detail || data?.message || "Prediction request failed.";
    throw new Error(message);
  }

  return data;
}
