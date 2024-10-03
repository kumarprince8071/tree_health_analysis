// src/App.js
import React from 'react';
import ImageUpload from './Componenets/ImageUpload';
import ChatInterface from './Componenets/ChatInterface';
import ExportData from './Componenets/ExportData';

function App() {
  return (
    <div className="bg-gray-100 min-h-screen flex flex-col">
      <h1 className="text-4xl font-bold text-center my-4">Tree Health Analysis</h1>
      <div className="flex-grow flex flex-col lg:flex-row">
        {/* Left side: Imagery Section */}
        <div className="w-full lg:w-2/3 lg:pr-4 flex flex-col">
          <ImageUpload />
        </div>
        {/* Right side: Chat Interface */}
        <div className="w-full lg:w-1/3 lg:pl-4 flex flex-col">
          <ChatInterface />
        </div>
      </div>
    </div>
  );
}

export default App;


