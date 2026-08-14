const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

export async function predictAge(imageFile) {
  const formData = new FormData();
  formData.append("image", imageFile);

  const response = await fetch(`${API_URL}/api/predict`, { method: "POST", body: formData });
  const data = await response.json().catch(() => ({}));

  if (!response.ok) throw new Error(data.message || "Prediction request failed.");

  return data;
}
