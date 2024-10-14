import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import torch
import math  
from torchvision import transforms
import uvicorn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import rasterio
from rasterio.transform import Affine
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import re
import cv2  # Added for image processing
import geopandas as gpd  # Added for geospatial data handling
from shapely.geometry import Polygon  # Added for creating polygons
from typing import Optional, Dict , Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv('MISTRAL_API_KEY')
if api_key is None:
    logger.warning("MISTRAL_API_KEY not set. Chat functionality will be limited.")

class ChatQuery(BaseModel):
    query: str
    tree_count: int = 0
    total_area: float = 0.0
    average_tree_area: Optional[float] = 0.0
    image_metadata: Optional[Dict[str, Any]] = {}

app = FastAPI()

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Consider restricting this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('exports', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Directory where your models are stored
MODEL_DIR = 'models'

# Cache to store loaded models to avoid reloading
loaded_models = {}

def get_detection_model(num_classes):
    try:
        # Create a ResNet-50 backbone with FPN
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)

        # Modify the conv1 layer to accept 4 input channels
        backbone.body.conv1 = torch.nn.Conv2d(
            in_channels=4,  # Set to 4 to match the input
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Create the MaskRCNN model
        model = MaskRCNN(backbone, num_classes=num_classes)
        return model
    except Exception as e:
        logger.error(f"Error creating detection model: {e}")
        raise RuntimeError("Failed to create the detection model.")

# Function to load model dynamically with caching
def load_model(model_name):
    if model_name in loaded_models:
        logger.info(f"Model '{model_name}' loaded from cache.")
        return loaded_models[model_name]

    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found in '{MODEL_DIR}' directory.")

    num_classes = 2  # Adjust as per your model (e.g., 1 class + background)
    model = get_detection_model(num_classes)

    try:
        # Load the state_dict
        state_dict = torch.load(model_path, map_location=device)
        logger.info(f"Loaded state_dict from '{model_path}'.")
    except Exception as e:
        logger.error(f"Error loading state_dict from '{model_path}': {e}")
        raise RuntimeError(f"Failed to load state_dict from '{model_path}'.")

    try:
        # Adjust the conv1 weights if necessary
        conv1_weights = state_dict['backbone.body.conv1.weight']
        if conv1_weights.shape[1] == 3:
            # Adjust weights to accept 4 channels by duplicating the first channel
            new_conv1_weights = torch.nn.Parameter(torch.zeros(64, 4, 7, 7))
            new_conv1_weights[:, :3, :, :] = conv1_weights
            new_conv1_weights[:, 3:, :, :] = conv1_weights[:, :1, :, :]  # Duplicate first channel
            state_dict['backbone.body.conv1.weight'] = new_conv1_weights
            logger.info("Adjusted conv1 weights to accept 4 channels.")
        elif conv1_weights.shape[1] == 4:
            # Weights are already compatible
            logger.info("conv1 weights already have 4 channels.")
        else:
            raise ValueError("Unexpected number of channels in conv1 weights.")

        # Load the modified state_dict
        model.load_state_dict(state_dict)
        logger.info(f"Loaded state_dict into model '{model_name}'.")
    except Exception as e:
        logger.error(f"Error processing conv1 weights for model '{model_name}': {e}")
        raise RuntimeError(f"Failed to process conv1 weights for model '{model_name}'.")

    # Modify the model's internal transforms to handle 4 channels
    try:
        model.transform = GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406, 0.5],  # Updated for 4 channels
            image_std=[0.229, 0.224, 0.225, 0.25]    # Updated for 4 channels
        )
        logger.info("Updated model's internal transforms to handle 4 channels.")
    except Exception as e:
        logger.error(f"Error updating model's transforms: {e}")
        raise RuntimeError("Failed to update model's internal transforms.")

    model.to(device)
    model.eval()
    loaded_models[model_name] = model
    logger.info(f"Model '{model_name}' loaded and cached successfully.")
    return model

# Endpoint to get the list of available models
@app.get("/models")
async def list_models():
    try:
        models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
        if not models:
            return JSONResponse(status_code=404, content={"message": "No models found."})
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to list models."})
    
# 2nd approach

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    model_name: str = Query(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    try:
        # Load the selected model
        detection_model = load_model(model_name)
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        return JSONResponse(status_code=404, content={"message": str(e)})
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to load the selected model."})

    try:
        # Save uploaded image
        image_path = os.path.join('uploads', image.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        logger.info(f"Saved uploaded image to '{image_path}'.")
    except Exception as e:
        logger.error(f"Error saving uploaded image: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to save uploaded image."})

    try:
        # Read the image using Rasterio
        with rasterio.open(image_path) as src:
            img_array = src.read()  # Read all bands
            crs = src.crs
            transform_raster = src.transform
            original_width = src.width
            original_height = src.height
            logger.info(f"Image CRS: {crs}")
            logger.info(f"Image transform: {transform_raster}")
            logger.info(f"Original image size: {original_width} x {original_height}")
        logger.info(f"Read image '{image_path}' with Rasterio.")
    except Exception as e:
        logger.error(f"Error reading image with Rasterio: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to read image file."})

    try:
        # Ensure the image has exactly 4 bands
        if img_array.shape[0] == 4:
            logger.info("Image has 4 bands.")
        elif img_array.shape[0] > 4:
            img_array = img_array[:4, :, :]  # Take the first 4 bands
            logger.info("Image has more than 4 bands. Truncated to first 4 bands.")
        else:
            # Handle images with fewer than 4 bands by duplicating channels
            channels_needed = 4 - img_array.shape[0]
            extra_channels = np.repeat(img_array[0:1, :, :], channels_needed, axis=0)
            img_array = np.concatenate((img_array, extra_channels), axis=0)
            logger.info(f"Image had fewer than 4 bands. Duplicated first band to make 4 bands.")

        # Convert NumPy array to PIL Image for compatibility with transforms
        # Transpose the array to (H, W, C)
        img_array_transposed = np.transpose(img_array, (1, 2, 0))
        # Convert to uint8 if necessary
        if img_array_transposed.dtype != np.uint8:
            img_min = img_array_transposed.min()
            img_max = img_array_transposed.max()
            img_array_transposed = ((img_array_transposed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            logger.info("Normalized image array to uint8.")
        img_pil = Image.fromarray(img_array_transposed)
        logger.info("Converted NumPy array to PIL Image.")

    except Exception as e:
        logger.error(f"Error processing image array: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to process image data."})

    try:
        # Image preprocessing transforms
        transform = transforms.ToTensor()
        logger.info("Defined image preprocessing transforms.")

        # Define tile size and overlap
        tile_size = 2048  # Adjust based on your GPU memory capacity
        overlap = 100  # Overlap between tiles to avoid edge effects

        # Calculate number of tiles in each dimension
        n_tiles_x = math.ceil((original_width - overlap) / (tile_size - overlap))
        n_tiles_y = math.ceil((original_height - overlap) / (tile_size - overlap))
        logger.info(f"Tiling image into {n_tiles_x} x {n_tiles_y} tiles.")

        # Initialize lists to store detections from all tiles
        all_polygons = []
        all_areas = []
        mask_overlay = np.zeros((original_height, original_width), dtype=np.uint8)

        # Loop over tiles
        for i in range(n_tiles_x):
            for j in range(n_tiles_y):
                # Calculate tile boundaries with overlap
                x_start = i * (tile_size - overlap)
                y_start = j * (tile_size - overlap)
                x_end = x_start + tile_size
                y_end = y_start + tile_size

                # Ensure the tile does not exceed image boundaries
                x_end = min(x_end, original_width)
                y_end = min(y_end, original_height)

                # Crop the tile from the image
                tile = img_pil.crop((x_start, y_start, x_end, y_end))

                # Apply transformations
                input_tensor = transform(tile).to(device)
                logger.info(f"Processing tile at position ({i}, {j}) with shape {input_tensor.shape}.")

                # Perform inference with no gradient calculation
                with torch.no_grad():
                    outputs = detection_model([input_tensor])[0]

                # Process outputs to get masks and scores
                scores = outputs['scores'].cpu().numpy()
                masks = outputs['masks'].cpu().numpy()  # shape: [N, 1, H, W]

                # Filter out detections based on the dynamic threshold
                selected_indices = scores >= threshold
                scores = scores[selected_indices]
                masks = masks[selected_indices]

                for k, mask in enumerate(masks):
                    # The mask is of shape [1, H, W], we need to convert it to [H, W]
                    mask = mask[0]
                    # Threshold the mask at 0.5
                    mask = (mask >= 0.5).astype(np.uint8)
                    # Resize mask to tile size if necessary
                    mask_height, mask_width = mask.shape
                    if mask.shape != (tile_size, tile_size):
                        mask = cv2.resize(mask, (tile.size[0], tile.size[1]), interpolation=cv2.INTER_NEAREST)

                    # Place the mask in the correct position in the full-size mask
                    mask_full = np.zeros((original_height, original_width), dtype=np.uint8)
                    mask_full[y_start:y_end, x_start:x_end] = mask

                    # Find contours
                    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        # Convert contour coordinates to x, y lists
                        contour = contour.reshape(-1, 2)
                        # Map pixel coordinates to spatial coordinates
                        spatial_coords = []
                        for x, y in contour:
                            row, col = y, x
                            lon, lat = rasterio.transform.xy(transform_raster, row, col)
                            spatial_coords.append((lon, lat))
                        # Create a polygon
                        if len(spatial_coords) >= 3:
                            polygon = Polygon(spatial_coords)
                            all_polygons.append(polygon)
                            all_areas.append(polygon.area)
                    # For visualization, add the mask to the overlay
                    mask_overlay = cv2.bitwise_or(mask_overlay, mask_full * 255)

        # Calculate total area
        total_area = sum(all_areas)
        tree_count = len(all_polygons)
        logger.info(f"Calculated total area from masks: {total_area}")

        # Create a GeoDataFrame with the polygons
        gdf = gpd.GeoDataFrame(geometry=all_polygons, crs=crs)
        # Save the GeoDataFrame to a file
        output_vector_path = os.path.join('outputs', f'detections_{image.filename}.geojson')
        gdf.to_file(output_vector_path, driver='GeoJSON')
        logger.info(f"Saved detections to '{output_vector_path}'.")

        # For visualization, overlay the masks on the image
        img_rgb = np.array(img_pil.convert('RGB'))  # Convert to RGB
        # Create a colored mask
        colored_mask = np.zeros_like(img_rgb)
        colored_mask[:, :, 1] = mask_overlay  # Highlight in green
        # Overlay the mask on the image
        overlayed_image = cv2.addWeighted(img_rgb, 1.0, colored_mask, 0.5, 0)

        # Save the output image
        output_image_path = os.path.join('outputs', f'overlay_{image.filename}.png')
        cv2.imwrite(output_image_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved overlay image to '{output_image_path}'.")

        # Update Code from ChatGPT
        image_width = original_width
        image_height = original_height
        crs_info = crs.to_string() if crs else "N/A"
        transform_info = str(transform_raster) if transform_raster else "N/A"

        # Read image bytes
        with open(output_image_path, "rb") as image_file:
            image_bytes = image_file.read()

        average_tree_area = total_area / tree_count if tree_count > 0 else 0.0


        # Return predicted data and image
        response_data = {
            "tree_count": tree_count,
            "total_area": total_area,
            "average_tree_area": average_tree_area,
            "image": image_bytes.hex(),  # Convert bytes to hex string
            "geojson_url": f"/outputs/detections_{image.filename}.geojson",  # Include geojson URL if needed
            "image_width": image_width,
            "image_height": image_height,
            "crs": crs_info,
            "transform": transform_info
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error processing model outputs: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to process model outputs."})
    
@app.post("/chat/")
async def chat_with_model(query: ChatQuery):
    try:
        user_query = query.query.lower()
        response = ""

        # Safeguard against division by zero
        average_tree_area = query.average_tree_area if query.average_tree_area else (query.total_area / query.tree_count if query.tree_count else 0.0)

        # Safeguard for image_metadata
        metadata = query.image_metadata if query.image_metadata else {}

        # Handle queries about tree count
        if any(keyword in user_query for keyword in ["how many trees", "tree count", "number of trees"]):
            response = f"There are {query.tree_count} trees detected in the satellite imagery."

        # Handle queries about total area
        elif any(keyword in user_query for keyword in ["total area", "area covered", "overall area"]):
            response = f"The total area covered by the detected trees is {query.total_area:.2f} square units."

        # Handle queries about average tree area
        elif any(keyword in user_query for keyword in ["average tree area", "average area per tree"]):
            response = f"The average area per tree is {average_tree_area:.2f} square units."

        # Handle queries about image metadata
        elif "image metadata" in user_query or "image info" in user_query:
            response = (
                f"Image Metadata:\n"
                f"Width: {metadata.get('width', 'N/A')} pixels\n"
                f"Height: {metadata.get('height', 'N/A')} pixels\n"
                f"CRS: {metadata.get('crs', 'N/A')}\n"
                f"Transform: {metadata.get('transform', 'N/A')}"
            )

        # Handle other queries using the Mistral AI API
        else:
            if not api_key:
                response = "Chat functionality is not available because the API key is not set."
            else:
                client = Mistral(api_key=api_key)
                model = "mistral-large-latest"
                chat_context = (
                    f"You are an AI assistant that provides information about GeoAI and Geographic Information Systems. "
                    f"Your responses should be informative and helpful. "
                    f"Your responses should be in the form of a natural language sentence. "
                    f"Use only the following information: {query.query}"
                    f"Your response should be pretty clear and concise. "
                )
                messages = [
                    {"role": "system", "content": chat_context},
                    {"role": "user", "content": query.query}
                ]
                chat_response = client.chat.complete(
                    model=model,
                    messages=messages,
                    temperature=0.9,

                )
                response = chat_response.choices[0].message.content.strip() 

        return {"query": query.query, "response": response}
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        return JSONResponse(status_code=500, content={"message": "An error occurred during chat processing."})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)