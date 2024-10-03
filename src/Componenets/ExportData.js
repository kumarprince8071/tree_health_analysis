// src/components/ExportData.js
import React from 'react';
import axios from 'axios';

const ExportData = () => {
  const handleExport = async (format) => {
    try {
      const response = await axios.get(`http://localhost:8000/export?format=${format}`, {
        responseType: 'blob',
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `exported_data.${format}`);
      document.body.appendChild(link);
      link.click();
    } catch (error) {
      console.error('Error exporting data:', error);
    }
  };

  return (
    <div className="p-4 bg-white shadow rounded">
      <h2 className="text-2xl font-bold mb-4">Export Data</h2>
      <div className="flex space-x-4">
        <button
          onClick={() => handleExport('shp')}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Export as SHP
        </button>
        <button
          onClick={() => handleExport('geojson')}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Export as GeoJSON
        </button>
      </div>
    </div>
  );
};

export default ExportData;
