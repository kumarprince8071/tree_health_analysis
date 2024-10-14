// src/services/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Use environment variable

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Fetch available models
export const fetchModels = async () => {
  try {
    const response = await api.get('/models');
    return response.data;
  } catch (error) {
    throw error;
  }
};

// Upload image and get predictions with progress tracking
export const uploadImage = async (formData, modelName, threshold = 0.5, onUploadProgress) => {
  try {
    const response = await api.post('/predict', formData, {
      params: { model_name: modelName, threshold },
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress,
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};

// Chat with the model
export const chatWithModel = async (
  query,
  treeCount,
  totalArea,
  averageTreeArea,
  imageMetadata,
  onChatProgress = null // Optional callback for chat progress
) => {
  try {
    const config = {};
    if (onChatProgress) {
      config.onUploadProgress = onChatProgress;
    }
    const response = await api.post(
      '/chat/',
      {
        query,
        tree_count: treeCount,
        total_area: totalArea,
        average_tree_area: averageTreeArea,
        image_metadata: imageMetadata,
      },
      config
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export default api;
