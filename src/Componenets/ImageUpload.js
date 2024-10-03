// src/components/ImageUpload.js
import React, { useState } from 'react';
import axios from 'axios';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';

const ImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictionImage, setPredictionImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPredictionImage(null);
    setModels([]);
    setSelectedModel('');
    fetchModels();
  };

  const fetchModels = () => {
    axios
      .get('http://localhost:8000/models')
      .then((response) => {
        setModels(response.data.models);
        if (response.data.models.length > 0) {
          setSelectedModel(response.data.models[0]);
        }
      })
      .catch((error) => {
        console.error('Error fetching models:', error);
      });
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    if (!selectedModel) {
      alert('Please select a model.');
      return;
    }
    setLoading(true);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post(
        `http://localhost:8000/predict?model_name=${encodeURIComponent(selectedModel)}`,
        formData,
        {
          responseType: 'blob',
        }
      );

      const imageUrl = URL.createObjectURL(response.data);
      setPredictionImage(imageUrl);
    } catch (error) {
      console.error('Error uploading image:', error);
    }

    setLoading(false);
  };

  return (
    <div className="p-4 bg-white shadow rounded mb-4">
      <h2 className="text-2xl font-bold mb-4">Image Upload and Prediction</h2>
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="block w-full text-sm text-gray-500
                   file:mr-4 file:py-2 file:px-4
                   file:rounded-full file:border-0
                   file:text-sm file:font-semibold
                   file:bg-violet-50 file:text-violet-700
                   hover:file:bg-violet-100"
      />

      {selectedFile && models.length > 0 && (
        <div className="mt-4">
          <label className="block text-gray-700 font-bold mb-2">Select Model:</label>
          <select
            value={selectedModel}
            onChange={handleModelChange}
            className="block w-full border border-gray-300 rounded py-2 px-3 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {models.map((model, index) => (
              <option key={index} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>
      )}

      <button
        onClick={handleUpload}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        disabled={!selectedFile || !selectedModel || loading}
      >
        {loading ? 'Processing...' : 'Upload and Predict'}
      </button>

      {predictionImage && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold mb-2">Prediction Result:</h3>
          <div
            className="border border-gray-300 overflow-hidden"
            style={{ width: '100%', height: '500px', position: 'relative' }}
          >
            <TransformWrapper
              defaultScale={1}
              defaultPositionX={0}
              defaultPositionY={0}
              wheel={{ step: 0.1 }}
              pan={{ disabled: false }}
            >
              {({ zoomIn, zoomOut, resetTransform, ...rest }) => (
                <>
                  <div className="tools" style={{ marginBottom: '10px' }}>
                    <button
                      onClick={() => zoomIn()}
                      className="px-2 py-1 bg-gray-200 mr-2 rounded"
                    >
                      Zoom In
                    </button>
                    <button
                      onClick={() => zoomOut()}
                      className="px-2 py-1 bg-gray-200 mr-2 rounded"
                    >
                      Zoom Out
                    </button>
                    <button
                      onClick={() => resetTransform()}
                      className="px-2 py-1 bg-gray-200 rounded"
                    >
                      Reset
                    </button>
                  </div>
                  <TransformComponent>
                    <img
                      src={predictionImage}
                      alt="Prediction Result"
                      style={{ width: '100%', height: 'auto' }}
                    />
                  </TransformComponent>
                </>
              )}
            </TransformWrapper>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
