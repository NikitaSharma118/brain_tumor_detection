import React, { useState } from "react";
import axios from "axios";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");
  const [gradCamImage, setGradCamImage] = useState("");
  const [explanation, setExplanation] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showModal, setShowModal] = useState(false);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setPrediction("");
    setConfidence("");
    setGradCamImage("");
    setExplanation("");
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedFile) {
      setError("⚠️ Please upload an image before predicting!");
      return;
    }

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      console.log(response.data); // debug

      if (response.data.success) {
        setPrediction(response.data.predicted_class);
        setConfidence(response.data.confidence);
        setGradCamImage(`data:image/jpeg;base64,${response.data.gradcam_image}`);
        setExplanation(response.data.explanation);
        setShowModal(true); // Show popup
      } else {
        setError("Prediction failed. Please try again.");
      }
    } catch (err) {
      console.error("Error connecting to backend:", err);
      setError("Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-6">
      <div className="bg-white shadow-lg rounded-2xl p-8 max-w-md w-full text-center">
        <h1 className="text-2xl font-bold text-gray-800 mb-4">
          Brain Tumor Detection
        </h1>
        <p className="text-gray-500 mb-4">
          Upload an MRI scan to predict tumor type and see Grad-CAM visualization.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="w-full border border-gray-300 rounded-lg p-2"
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition"
          >
            {loading ? "Analyzing..." : "Predict"}
          </button>
        </form>

        {error && <p className="text-red-500 mt-3">{error}</p>}
      </div>

      {/* Modal Popup */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl shadow-2xl p-6 w-96 relative">
            <button
              onClick={() => setShowModal(false)}
              className="absolute top-2 right-3 text-gray-500 hover:text-black text-xl"
            >
              ✖
            </button>

            <h2 className="text-xl font-bold text-center mb-4 text-blue-700">
              Prediction Result
            </h2>

            <p className="text-lg font-semibold text-gray-800">
              Predicted: {prediction.toUpperCase()}
            </p>
            <p className="text-gray-600 mb-3">Confidence: {confidence}</p>

            {gradCamImage && (
              <img
                src={gradCamImage}
                alt="Grad-CAM"
                className="rounded-lg shadow-md w-full mb-3"
              />
            )}

            {explanation && (
              <p className="text-sm text-gray-700 leading-relaxed mt-2 border-t pt-2">
                {explanation}
              </p>
            )}

            <button
              onClick={() => setShowModal(false)}
              className="mt-5 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition w-full"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
