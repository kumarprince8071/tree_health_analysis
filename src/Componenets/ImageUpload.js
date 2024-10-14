// // src/components/ImageUpload.js

// import React, { useState } from 'react';
// import axios from 'axios';
// import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';

// const ImageUpload = ({ setTreeCount, setTotalArea }) => {
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [predictionImage, setPredictionImage] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [progress, setProgress] = useState(0); // State for progress
//   const [models, setModels] = useState([]);
//   const [selectedModel, setSelectedModel] = useState('');

//   const handleFileChange = (event) => {
//     setSelectedFile(event.target.files[0]);
//     setPredictionImage(null);
//     setModels([]);
//     setSelectedModel('');
//     fetchModels();
//   };

//   const fetchModels = () => {
//     axios
//       .get('http://localhost:8000/models')
//       .then((response) => {
//         setModels(response.data.models);
//         if (response.data.models.length > 0) {
//           setSelectedModel(response.data.models[0]);
//         }
//       })
//       .catch((error) => {
//         console.error('Error fetching models:', error);
//       });
//   };

//   const handleModelChange = (event) => {
//     setSelectedModel(event.target.value);
//   };

//   const handleUpload = async () => {
//     if (!selectedFile) return;
//     if (!selectedModel) {
//       alert('Please select a model.');
//       return;
//     }
//     setLoading(true);
//     setProgress(0); // Reset progress

//     const formData = new FormData();
//     formData.append('image', selectedFile);

//     try {
//       // Start incrementing progress
//       incrementProgress();

//       const response = await axios.post(
//         `http://localhost:8000/predict?model_name=${encodeURIComponent(selectedModel)}`,
//         formData,
//         {
//           timeout: 600000, // Increase timeout if necessary
//           onUploadProgress: (progressEvent) => {
//             const percentCompleted = Math.round(
//               (progressEvent.loaded * 100) / progressEvent.total
//             );
//             setProgress(percentCompleted * 0.3); // Upload progress up to 30%
//           },
//         }
//       );

//       const data = response.data;
//       setTreeCount(data.tree_count);
//       setTotalArea(data.total_area);

//       // Convert hex string back to bytes
//       const imageBytes = new Uint8Array(
//         data.image.match(/.{1,2}/g).map((byte) => parseInt(byte, 16))
//       );
//       const blob = new Blob([imageBytes], { type: 'image/png' });
//       const imageUrl = URL.createObjectURL(blob);
//       setPredictionImage(imageUrl);

//       // Set progress to 100% when done
//       setProgress(100);
//     } catch (error) {
//       console.error('Error uploading image:', error);
//       alert('An error occurred during processing.');
//     } finally {
//       setLoading(false);
//     }
//   };

//   // Function to increment progress up to 90%
//   const incrementProgress = () => {
//     setProgress((prevProgress) => {
//       if (prevProgress >= 90) {
//         return prevProgress;
//       } else {
//         return prevProgress + 1;
//       }
//     });

//     if (progress < 90) {
//       setTimeout(incrementProgress, 1000); // Adjust interval as needed
//     }
//   };

//   return (
//     <div className="p-4 bg-white shadow rounded mb-4">
//       <h2 className="text-2xl font-bold mb-4">Image Upload and Prediction</h2>
//       <input
//         type="file"
//         accept="image/*"
//         onChange={handleFileChange}
//         className="block w-full text-sm text-gray-500
//                          file:mr-4 file:py-2 file:px-4
//                          file:rounded-full file:border-0
//                          file:text-sm file:font-semibold
//                          file:bg-violet-50 file:text-violet-700
//                          hover:file:bg-violet-100"
//       />

//       {selectedFile && models.length > 0 && (
//         <div className="mt-4">
//           <label className="block text-gray-700 font-bold mb-2">Select Model:</label>
//           <select
//             value={selectedModel}
//             onChange={handleModelChange}
//             className="block w-full border border-gray-300 rounded py-2 px-3 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-500"
//           >
//             {models.map((model, index) => (
//               <option key={index} value={model}>
//                 {model}
//               </option>
//             ))}
//           </select>
//         </div>
//       )}

//       <button
//         onClick={handleUpload}
//         className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
//         disabled={!selectedFile || !selectedModel || loading}
//       >
//         {loading ? 'Processing...' : 'Upload and Predict'}
//       </button>

//       {/* Progress Bar */}
//       {loading && (
//         <div className="mt-4">
//           <div className="w-full bg-gray-200 rounded-full h-2.5">
//             <div
//               className="bg-blue-600 h-2.5 rounded-full"
//               style={{ width: `${progress}%` }}
//             ></div>
//           </div>
//           <p className="text-center mt-2">{Math.round(progress)}%</p>
//         </div>
//       )}

//       {predictionImage && (
//         <div className="mt-6">
//           <h3 className="text-xl font-semibold mb-2">Prediction Result:</h3>
//           <div
//             className="border border-gray-300"
//             style={{ width: '100%', height: 'auto', position: 'relative' }}
//           >
//             <TransformWrapper
//               defaultScale={1}
//               wheel={{ step: 0.1 }}
//               pan={{ disabled: false }}
//             >
//               {({ zoomIn, zoomOut, resetTransform, ...rest }) => (
//                 <>
//                   <div className="tools" style={{ marginBottom: '10px' }}>
//                     <button
//                       onClick={() => zoomIn()}
//                       className="px-2 py-1 bg-gray-200 mr-2 rounded"
//                     >
//                       Zoom In
//                     </button>
//                     <button
//                       onClick={() => zoomOut()}
//                       className="px-2 py-1 bg-gray-200 mr-2 rounded"
//                     >
//                       Zoom Out
//                     </button>
//                     <button
//                       onClick={() => resetTransform()}
//                       className="px-2 py-1 bg-gray-200 rounded"
//                     >
//                       Reset
//                     </button>
//                   </div>
//                   <TransformComponent>
//                     <img
//                       src={predictionImage}
//                       alt="Prediction Result"
//                       style={{ width: '100%', height: 'auto', maxWidth: '100%' }}
//                     />
//                   </TransformComponent>
//                 </>
//               )}
//             </TransformWrapper>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default ImageUpload;
// src/components/ImageUpload.js

// import React, { useState } from 'react';
// import axios from 'axios';
// import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';

// const ImageUpload = ({ setTreeCount, setTotalArea, setAverageTreeArea, setImageMetadata }) => {
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [predictionImage, setPredictionImage] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [progress, setProgress] = useState(0);
//   const [models, setModels] = useState([]);
//   const [selectedModel, setSelectedModel] = useState('');
//   const [geojsonUrl, setGeojsonUrl] = useState(null);
//   const [threshold, setThreshold] = useState(0.5);

//   const handleFileChange = (event) => {
//     setSelectedFile(event.target.files[0]);
//     setPredictionImage(null);
//     setModels([]);
//     setSelectedModel('');
//     setGeojsonUrl(null);
//     fetchModels();
//   };

//   const fetchModels = () => {
//     axios
//       .get('http://localhost:8000/models')
//       .then((response) => {
//         setModels(response.data.models);
//         if (response.data.models.length > 0) {
//           setSelectedModel(response.data.models[0]);
//         }
//       })
//       .catch((error) => {
//         console.error('Error fetching models:', error);
//       });
//   };

//   const handleModelChange = (event) => {
//     setSelectedModel(event.target.value);
//   };

//   const handleUpload = async () => {
//     if (!selectedFile) return;
//     if (!selectedModel) {
//       alert('Please select a model.');
//       return;
//     }
//     setLoading(true);
//     setProgress(0);

//     const formData = new FormData();
//     formData.append('image', selectedFile);

//     try {
//       // Start incrementing progress
//       incrementProgress();

//       const response = await axios.post(
//         `http://localhost:8000/predict?model_name=${encodeURIComponent(selectedModel)}&threshold=${threshold}`,
//         formData,
//         {
//           timeout: 600000,
//           onUploadProgress: (progressEvent) => {
//             const percentCompleted = Math.round(
//               (progressEvent.loaded * 100) / progressEvent.total
//             );
//             setProgress(percentCompleted * 0.3);
//           },
//         }
//       );

//       const data = response.data;
//       setTreeCount(data.tree_count);
//       setTotalArea(data.total_area);
//       setAverageTreeArea(data.average_tree_area);
//       setImageMetadata(data.image_metadata);

//       // Convert hex string back to bytes
//       const imageBytes = new Uint8Array(
//         data.image.match(/.{1,2}/g).map((byte) => parseInt(byte, 16))
//       );
//       const blob = new Blob([imageBytes], { type: 'image/png' });
//       const imageUrl = URL.createObjectURL(blob);
//       setPredictionImage(imageUrl);

//       // Set progress to 100% when done
//       setProgress(100);

//       // Set the GeoJSON URL
//       if (data.geojson_url) {
//         setGeojsonUrl(`http://localhost:8000${data.geojson_url}`);
//       } else {
//         setGeojsonUrl(null);
//       }
//     } catch (error) {
//       console.error('Error uploading image:', error);
//       alert('An error occurred during processing.');
//     } finally {
//       setLoading(false);
//     }
//   };

//   // Function to increment progress up to 90%
//   const incrementProgress = () => {
//     setProgress((prevProgress) => {
//       if (prevProgress >= 90) {
//         return prevProgress;
//       } else {
//         return prevProgress + 1;
//       }
//     });

//     if (progress < 90) {
//       setTimeout(incrementProgress, 6000);
//     }
//   };

//   return (
//     <div className="p-4 bg-white shadow rounded mb-4">
//       <h2 className="text-2xl font-bold mb-4">Image Upload and Prediction</h2>
//       <input
//         type="file"
//         accept="image/*"
//         onChange={handleFileChange}
//         className="block w-full text-sm text-gray-500
//                                file:mr-4 file:py-2 file:px-4
//                                file:rounded-full file:border-0
//                                file:text-sm file:font-semibold
//                                file:bg-violet-50 file:text-violet-700
//                                hover:file:bg-violet-100"
//       />

//       {selectedFile && models.length > 0 && (
//         <div className="mt-4">
//           <label className="block text-gray-700 font-bold mb-2">Select Model:</label>
//           <select
//             value={selectedModel}
//             onChange={handleModelChange}
//             className="block w-full border border-gray-300 rounded py-2 px-3 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-500"
//           >
//             {models.map((model, index) => (
//               <option key={index} value={model}>
//                 {model}
//               </option>
//             ))}
//           </select>
//         </div>
//       )}

//       {/* Confidence Threshold Slider */}
//       <div className="mt-4">
//         <label className="block text-gray-700 font-bold mb-2">
//           Confidence Threshold: {threshold}
//         </label>
//         <input
//           type="range"
//           min="0"
//           max="1"
//           step="0.01"
//           value={threshold}
//           onChange={(e) => setThreshold(parseFloat(e.target.value))}
//           className="w-full"
//         />
//       </div>

//       <button
//         onClick={handleUpload}
//         className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
//         disabled={!selectedFile || !selectedModel || loading}
//       >
//         {loading ? 'Processing...' : 'Upload and Predict'}
//       </button>

//       {/* Progress Bar */}
//       {loading && (
//         <div className="mt-4">
//           <div className="w-full bg-gray-200 rounded-full h-2.5">
//             <div
//               className="bg-blue-600 h-2.5 rounded-full"
//               style={{ width: `${progress}%` }}
//             ></div>
//           </div>
//           <p className="text-center mt-2">{Math.round(progress)}%</p>
//         </div>
//       )}

//       {predictionImage && (
//         <div className="mt-6">
//           <h3 className="text-xl font-semibold mb-2">Prediction Result:</h3>
//           <div
//             className="border border-gray-300"
//             style={{ width: '100%', height: 'auto', position: 'relative' }}
//           >
//             <TransformWrapper
//               defaultScale={1}
//               wheel={{ step: 0.1 }}
//               pan={{ disabled: false }}
//             >
//               {({ zoomIn, zoomOut, resetTransform, ...rest }) => (
//                 <>
//                   <div className="tools" style={{ marginBottom: '10px' }}>
//                     <button
//                       onClick={() => zoomIn()}
//                       className="px-2 py-1 bg-gray-200 mr-2 rounded"
//                     >
//                       Zoom In
//                     </button>
//                     <button
//                       onClick={() => zoomOut()}
//                       className="px-2 py-1 bg-gray-200 mr-2 rounded"
//                     >
//                       Zoom Out
//                     </button>
//                     <button
//                       onClick={() => resetTransform()}
//                       className="px-2 py-1 bg-gray-200 rounded"
//                     >
//                       Reset
//                     </button>
//                   </div>
//                   <TransformComponent>
//                     <img
//                       src={predictionImage}
//                       alt="Prediction Result"
//                       style={{ width: '100%', height: 'auto', maxWidth: '100%' }}
//                     />
//                   </TransformComponent>
//                 </>
//               )}
//             </TransformWrapper>
//           </div>

//           {/* Download GeoJSON button */}
//           {geojsonUrl && (
//             <div className="mt-4">
//               <a
//                 href={geojsonUrl}
//                 download={`detections_${selectedFile.name}.geojson`}
//                 className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
//               >
//                 Download GeoJSON
//               </a>
//             </div>
//           )}
//         </div>
//       )}
//     </div>
//   );
// };

// export default ImageUpload;

// Tesing Approach 

import React, { useState } from 'react';
import axios from 'axios';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';

const ImageUpload = ({ setTreeCount, setTotalArea, setAverageTreeArea, setImageMetadata }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictionImage, setPredictionImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [geojsonUrl, setGeojsonUrl] = useState(null);
  const [threshold, setThreshold] = useState(0.5);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPredictionImage(null);
    setModels([]);
    setSelectedModel('');
    setGeojsonUrl(null);
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
        `http://localhost:8000/predict?model_name=${encodeURIComponent(
          selectedModel
        )}&threshold=${threshold}`,
        formData,
        {
          timeout: 600000,
        }
      );

      const data = response.data;
      setTreeCount(data.tree_count);
      setTotalArea(data.total_area);
      setAverageTreeArea(data.average_tree_area);
      setImageMetadata(data.image_metadata);

      // Set the prediction image URL
      const imageUrl = `http://localhost:8000${data.overlay_image_url}`;
      setPredictionImage(imageUrl);

      // Set the GeoJSON URL
      if (data.geojson_url) {
        setGeojsonUrl(`http://localhost:8000${data.geojson_url}`);
      } else {
        setGeojsonUrl(null);
      }
    } catch (error) {
      console.error('Error uploading image:', error);
      if (error.response && error.response.data && error.response.data.message) {
        alert(`An error occurred: ${error.response.data.message}`);
      } else {
        alert('An error occurred during processing.');
      }
    } finally {
      setLoading(false);
    }
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

      {/* Confidence Threshold Slider */}
      <div className="mt-4">
        <label className="block text-gray-700 font-bold mb-2">
          Confidence Threshold: {threshold}
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-full"
        />
      </div>

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
            className="border border-gray-300"
            style={{ width: '100%', height: 'auto', position: 'relative' }}
          >
            <TransformWrapper
              defaultScale={1}
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
                      style={{ width: '100%', height: 'auto', maxWidth: '100%' }}
                    />
                  </TransformComponent>
                </>
              )}
            </TransformWrapper>
          </div>

          {/* Download GeoJSON button */}
          {geojsonUrl && (
            <div className="mt-4">
              <a
                href={geojsonUrl}
                download={`detections_${selectedFile.name}.geojson`}
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
              >
                Download GeoJSON
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ImageUpload;

