import React, { useState } from 'react';

const FLASK_API_URL = 'http://127.0.0.1:5000/predict';

const UploadCard = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [gradCamImage, setGradCamImage] = useState(null);
    const [explanation, setExplanation] = useState(null);
    const [showModal, setShowModal] = useState(false); // added for popup

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
        setResult(null);
        setError(null);
        setGradCamImage(null);
        setExplanation(null);
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select a scan file (.jpg, .jpeg, .png, etc.) to upload.');
            return;
        }

        setIsLoading(true);
        setError(null);
        setResult(null);
        setGradCamImage(null);
        setExplanation(null);

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch(FLASK_API_URL, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server returned status ${response.status}: ${errorText}`);
            }

            const data = await response.json();

            if (data.success) {
                setResult({
                    predicted_class: data.predicted_class,
                    confidence: data.confidence,
                    explanation: data.explanation,
                });
                if (data.gradcam_image) {
                    setGradCamImage(`data:image/jpeg;base64,${data.gradcam_image}`);
                }
                setShowModal(true); // open popup when analysis done
            } else {
                setError(data.message || 'Analysis failed. Please try again.');
            }
        } catch (err) {
            console.error('Network or server error:', err);
            setError('Could not connect to the analysis server. Please ensure the backend is running at http://127.0.0.1:5000 and that CORS is enabled.');
        } finally {
            setIsLoading(false);
        }
    };

    const cardClass = "bg-gray-800 p-8 rounded-xl shadow-2xl shadow-sky-900/50 transition-all duration-300";
    const buttonClass = "w-full py-3 px-6 rounded-lg font-semibold transition-colors duration-300";
    const fileInputClass = "block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-sky-500/10 file:text-sky-400 hover:file:bg-sky-500/20 cursor-pointer";

    return (
        <section id="process" className="bg-[#0A0C16] flex flex-col items-center justify-center py-20">
            
            <div className={`max-w-xl max-w-sm ${cardClass}`}>
                <h3 className="text-3xl font-bold text-white mb-6 text-center">
                    Mind AI Analysis
                </h3>

                <div className="mb-6">
                    <label className="text-gray-400 block mb-2">Select Brain Scan Image (JPG, JPEG, PNG)</label>
                    <input 
                        type="file" 
                        onChange={handleFileChange} 
                        className={fileInputClass}
                        accept=".jpg,.jpeg,.png" 
                    />
                    {selectedFile && (
                        <p className="mt-2 text-sm text-sky-400">Selected: {selectedFile.name}</p>
                    )}
                </div>

                <button
                    onClick={handleUpload}
                    disabled={!selectedFile || isLoading}
                    className={`${buttonClass} ${
                        isLoading 
                            ? 'bg-gray-600 cursor-not-allowed' 
                            : 'bg-sky-600 hover:bg-sky-500 text-white'
                    }`}
                >
                    {isLoading ? 'Analyzing Scan...' : 'Analyze with Mind AI'}
                </button>

                <div className="mt-8">
                    {isLoading && (
                        <div className="flex items-center justify-center space-x-2 text-sky-400">
                            <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span>Processing...</span>
                        </div>
                    )}

                    {error && (
                        <p className="text-red-400 text-center font-semibold">{error}</p>
                    )}
                </div>
            </div>

            {/* Scrollable Popup for Results */}
            {showModal && result && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
                    style={{ overflowY: "auto", padding: "2rem" }}
                >
                    <div
                        className="bg-gray-900 rounded-2xl shadow-2xl p-6 w-full max-w-md relative"
                        style={{
                            maxHeight: "80vh",
                            overflowY: "auto",
                            display: "flex",
                            flexDirection: "column",
                            justifyContent: "flex-start",
                        }}
                    >
                        <button
                            onClick={() => setShowModal(false)}
                            className="absolute top-3 right-4 text-gray-500 hover:text-black text-xl font-bold"
                        >
                            ✕
                        </button>

                        <h2 className="text-xl font-bold text-center mb-4 text-cyan-400">
                            Prediction Result
                        </h2>

                        <p className="text-lg font-semibold text-dimWhite text-center">
                            Predicted: {result.predicted_class?.toUpperCase()}
                        </p>
                        <p className="text-dimWhite mb-3 text-center">
                            Confidence: {result.confidence}
                        </p>

                        {gradCamImage && (
                            <img
                                src={gradCamImage}
                                alt="Grad-CAM"
                                className="rounded-lg shadow-md w-full mb-3"
                                style={{ objectFit: "contain", maxHeight: "300px" }}
                            />
                        )}

                        {result.explanation && (
                            <p className="text-sm text-dimWhite leading-relaxed mt-2 border-t pt-2">
                                {result.explanation}
                            </p>
                        )}

                        <button
                            onClick={() => setShowModal(false)}
                            className="mt-5 bg-cyan-400 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition w-full"
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}
        </section>
    );
};

export default UploadCard;
