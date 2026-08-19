function ResultCard({ result }) {
  const timestamp = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(result.timestamp);

  return (
    <section className="result" aria-live="polite">
      <div className="success-check" aria-hidden="true">✓</div>
      <div>
        <p className="result-label">Prediction Result</p>
        <div className="result-values">
          <p>🎂 <strong>Predicted Age:</strong> {result.age} Years</p>
          <p>🎯 <strong>Confidence:</strong> <span className="confidence-badge">{result.confidence}%</span></p>
          <p>⏱️ <strong>Inference Time:</strong> {result.inference_time ?? 0.0}s</p>
        </div>
        <p className="timestamp">Analyzed {timestamp}</p>
      </div>
    </section>
  );
}

export default ResultCard;
