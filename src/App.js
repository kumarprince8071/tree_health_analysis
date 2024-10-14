// import React, { useState } from 'react';
// import ImageUpload from './Componenets/ImageUpload';
// import ChatInterface from './Componenets/ChatInterface';
// import ExportData from './Componenets/ExportData';

// function App() {
//   const [treeCount, setTreeCount] = useState(0);
//   const [totalArea, setTotalArea] = useState(0.0);

//   return (
//     <div className="bg-gray-100 min-h-screen flex flex-col">
//       <h1 className="text-4xl font-bold text-center my-4">Tree Health Analysis</h1>
//       <div className="flex-grow flex flex-col lg:flex-row">
//         {/* Left side: Imagery Section */}
//         <div className="w-full lg:w-2/3 lg:pr-4 flex flex-col">
//           <ImageUpload setTreeCount={setTreeCount} setTotalArea={setTotalArea} />
//         </div>
//         {/* Right side: Chat Interface */}
//         <div className="w-full lg:w-1/3 lg:pl-4 flex flex-col">
//           <ChatInterface treeCount={treeCount} totalArea={totalArea} />
//         </div>
//       </div>
//     </div>
//   );
// }

// export default App;

// src/App.jsx
import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import ImageViewer from './components/ImageViewer';
import Chat from './components/Chat';

const App = () => {
  const [predictionData, setPredictionData] = useState(null);

  const handleUploadSuccess = (data) => {
    setPredictionData(data);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="container mx-auto">
        <div className="flex flex-col lg:flex-row space-y-4 lg:space-y-0 lg:space-x-4">
          {/* Left Section: Image Upload and Viewer */}
          <div className="lg:w-2/3 w-full flex flex-col space-y-4">
            <ImageUploader onUploadSuccess={handleUploadSuccess} />
            <div className="h-96 bg-white shadow rounded">
              <ImageViewer imageData={predictionData?.image} />
            </div>
          </div>

          {/* Right Section: Chat */}
          <div className="lg:w-1/3 w-full flex flex-col">
            <Chat
              treeCount={predictionData?.tree_count || 0}
              totalArea={predictionData?.total_area || 0.0}
              averageTreeArea={predictionData?.total_area && predictionData?.tree_count
                ? predictionData.total_area / predictionData.tree_count
                : 0.0}
              imageMetadata={{
                width: predictionData?.image_width || 'N/A',
                height: predictionData?.image_height || 'N/A',
                crs: predictionData?.crs || 'N/A',
                transform: predictionData?.transform || 'N/A',
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
