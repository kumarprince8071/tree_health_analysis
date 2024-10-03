# import os
# import logging
# from fastapi import FastAPI, File, UploadFile, HTTPException, Query
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import shutil
# import torch
# from torchvision import transforms
# import uvicorn
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# from PIL import Image
# from torchvision.models.detection import MaskRCNN
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# import geopandas as gpd
# from shapely.geometry import box
# import rasterio
# from rasterio.transform import Affine
# import numpy as np
# from dotenv import load_dotenv
# from mistralai import Mistral

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# load_dotenv()
# api_key = os.getenv('MISTRAL_API_KEY')
# if api_key is None:
#     raise ValueError("Please set the MISTRAL_API_KEY environment variable.")

# predicted_data = {
#     "tree_count": None,
#     "total_area": None,
# }

# class ChatQuery(BaseModel):
#     query: str

# app = FastAPI()

# # CORS settings
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# os.makedirs('uploads', exist_ok=True)
# os.makedirs('outputs', exist_ok=True)
# os.makedirs('exports', exist_ok=True)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Directory where your models are stored
# MODEL_DIR = 'models'

# def get_detection_model(num_classes):
#     try:
#         # Create a ResNet-50 backbone with FPN
#         backbone = resnet_fpn_backbone('resnet50', pretrained=False)
        
#         # Modify the conv1 layer to accept 4 channels
#         backbone.body.conv1 = torch.nn.Conv2d(
#             in_channels=4,  # Change to 4 channels
#             out_channels=64,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False
#         )
        
#         # Create the model
#         model = MaskRCNN(backbone, num_classes=num_classes)
#         return model
#     except Exception as e:
#         logging.error(f"Error creating detection model: {e}")
#         raise RuntimeError("Failed to create the detection model.")

# # def get_detection_model(num_classes):
# #     try:
# #         # Create a ResNet-50 backbone with FPN
# #         backbone = resnet_fpn_backbone('resnet50', pretrained=False)
# #         # Create the model
# #         model = MaskRCNN(backbone, num_classes=num_classes)
# #         return model
# #     except Exception as e:
# #         logging.error(f"Error creating detection model: {e}")
# #         raise RuntimeError("Failed to create the detection model.")

# # Function to load model dynamically
# def load_model(model_name):
#     model_path = os.path.join(MODEL_DIR, model_name)
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model '{model_name}' not found.")

#     num_classes = 2  # Adjust as per your model
#     model = get_detection_model(num_classes)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#     return model

# # Endpoint to get the list of available models
# @app.get("/models")
# async def list_models():
#     try:
#         models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
#         return {"models": models}
#     except Exception as e:
#         logging.error(f"Error listing models: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to list models."})

# # Image preprocessing transforms
# transform = transforms.Compose([
#     # Since we're reading images as NumPy arrays, we can normalize manually if needed
#     # Add normalization if required
# ])

# # Helper function to create shapely box
# def box_shape(coords):
#     minx, miny, maxx, maxy = coords
#     return box(minx, miny, maxx, maxy)

# # @app.post("/predict")
# # async def predict(image: UploadFile = File(...), model_name: str = Query(...)):
# #     try:
# #         # Load the selected model
# #         detection_model = load_model(model_name)
# #     except FileNotFoundError as e:
# #         logging.error(f"Model not found: {e}")
# #         return JSONResponse(status_code=404, content={"message": str(e)})
# #     except Exception as e:
# #         logging.error(f"Error loading model: {e}")
# #         return JSONResponse(status_code=500, content={"message": "Failed to load the selected model."})

# #     try:
# #         # Save uploaded image
# #         image_path = os.path.join('uploads', image.filename)
# #         with open(image_path, "wb") as buffer:
# #             shutil.copyfileobj(image.file, buffer)
# #     except Exception as e:
# #         logging.error(f"Error saving uploaded image: {e}")
# #         return JSONResponse(status_code=500, content={"message": "Failed to save uploaded image."})

# #     try:
# #         # Read the image using Rasterio
# #         with rasterio.open(image_path) as src:
# #             img_array = src.read()  # Read all bands
# #             crs = src.crs
# #             transform_raster = src.transform
# #     except Exception as e:
# #         logging.error(f"Error reading image with Rasterio: {e}")
# #         return JSONResponse(status_code=500, content={"message": "Failed to read image file."})

# #     try:
# #         # Handle images with more than 3 bands
# #         if img_array.shape[0] >= 3:
# #             img_array = img_array[:3, :, :]  # Take the first 3 bands
# #         else:
# #             # Handle images with fewer than 3 bands by repeating channels
# #             img_array = np.tile(img_array, (3, 1, 1))

# #         # Convert NumPy array to PIL Image for compatibility with transforms
# #         img_pil = Image.fromarray(np.transpose(img_array, (1, 2, 0)).astype('uint8'), 'RGB')
# #     except Exception as e:
# #         logging.error(f"Error processing image array: {e}")
# #         return JSONResponse(status_code=500, content={"message": "Failed to process image data."})

# #     try:
# #         # Image preprocessing transforms
# #         transform = transforms.Compose([
# #             transforms.ToTensor(),
# #             transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Replace with your training means
# #                                  std=[0.229, 0.224, 0.225])   # Replace with your training stds
# #         ])

# #         # Apply transformations
# #         input_tensor = transform(img_pil).unsqueeze(0).to(device)
# #     except Exception as e:
# #         logging.error(f"Error during image preprocessing: {e}")
# #         return JSONResponse(status_code=500, content={"message": "Failed to preprocess image."})

# #     try:
# #         # Perform inference
# #         with torch.no_grad():
# #             outputs = detection_model(input_tensor)[0]  # Assuming the output is a dict
# #     except Exception as e:
# #         logging.error(f"Error during model inference: {e}")
# #         return JSONResponse(status_code=500, content={"message": "Failed to perform model inference."})

# #     try:
# #         # Process outputs to get bounding boxes and scores
# #         boxes = outputs['boxes'].cpu().numpy()
# #         scores = outputs['scores'].cpu().numpy()

# #         # Filter out low confidence detections
# #         threshold = 0.7
# #         selected_indices = scores >= threshold
# #         boxes = boxes[selected_indices]
# #         scores = scores[selected_indices]

# #         # Update predicted data
# #         predicted_data['tree_count'] = len(boxes)
# #         # For total_area, you might need to calculate based on boxes and image metadata

# #         # For visualization, ensure the image is in the correct format
# #         img = np.array(img_pil)

# #         # Draw bounding boxes on the image
# #         fig, ax = plt.subplots(1, figsize=(12, 12))
# #         ax.imshow(img)

# #         for box in boxes:
# #             x1, y1, x2, y2 = box
# #             width = x2 - x1
# #             height = y2 - y1
# #             rect = plt.Rectangle((x1, y1), width, height,
# #                                  linewidth=2, edgecolor='r', facecolor='none')
# #             ax.add_patch(rect)

# #         # Save the output image
# #         output_image_path = os.path.join('outputs', 'overlay_image.png')
# #         plt.axis('off')
# #         plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=300)
# #         plt.close(fig)
# #         return FileResponse(output_image_path, media_type="image/png")
# #     except Exception as e:
# #         logging.error(f"Error processing model outputs: {e}")
# #         return JSONResponse(status_code=500, content={"message": "Failed to process model outputs."})

# @app.post("/predict")
# async def predict(image: UploadFile = File(...), model_name: str = Query(...)):
#     try:
#         # Load the selected model
#         detection_model = load_model(model_name)
#     except FileNotFoundError as e:
#         logging.error(f"Model not found: {e}")
#         return JSONResponse(status_code=404, content={"message": str(e)})
#     except Exception as e:
#         logging.error(f"Error loading model: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to load the selected model."})

#     try:
#         # Save uploaded image
#         image_path = os.path.join('uploads', image.filename)
#         with open(image_path, "wb") as buffer:
#             shutil.copyfileobj(image.file, buffer)
#     except Exception as e:
#         logging.error(f"Error saving uploaded image: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to save uploaded image."})

#     try:
#         # Read the image using Rasterio
#         with rasterio.open(image_path) as src:
#             img_array = src.read()  # Read all bands
#             crs = src.crs
#             transform_raster = src.transform
#     except Exception as e:
#         logging.error(f"Error reading image with Rasterio: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to read image file."})

#     try:
#         # Ensure the image has at least 4 bands
#         if img_array.shape[0] >= 4:
#             img_array = img_array[:4, :, :]  # Take the first 4 bands
#         else:
#             raise ValueError("Input image must have at least 4 channels.")

#         # Convert NumPy array to PIL Image for compatibility with transforms
#         img_pil = Image.fromarray(np.transpose(img_array, (1, 2, 0)).astype('uint8'), 'RGBA')
#     except Exception as e:
#         logging.error(f"Error processing image array: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to process image data."})

#     try:
#         # Image preprocessing transforms
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406, 0.5],  # Replace with your training means
#                 std=[0.229, 0.224, 0.225, 0.25]   # Replace with your training stds
#             )
#         ])

#         # Apply transformations
#         input_tensor = transform(img_pil).unsqueeze(0).to(device)
#     except Exception as e:
#         logging.error(f"Error during image preprocessing: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to preprocess image."})

#     try:
#         # Perform inference
#         with torch.no_grad():
#             outputs = detection_model(input_tensor)[0]  # Assuming the output is a dict
#     except Exception as e:
#         logging.error(f"Error during model inference: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to perform model inference."})

# # Endpoint for exporting data
# @app.get("/export")
# async def export_data(format: str):
#     try:
#         if format not in ['shp', 'geojson']:
#             return JSONResponse(status_code=400, content={"message": "Invalid format"})
#         file_path = os.path.join('exports', f'exported_data.{format}')
#         if os.path.exists(file_path):
#             media_type = "application/octet-stream"
#             return FileResponse(file_path, media_type=media_type, filename=f'exported_data.{format}')
#         else:
#             return JSONResponse(status_code=404, content={"message": "File not found"})
#     except Exception as e:
#         logging.error(f"Error in /export endpoint: {e}")
#         return JSONResponse(status_code=500, content={"message": "An error occurred during data export."})

# @app.post("/chat/")
# async def chat_with_model(query: ChatQuery):
#     try:
#         user_query = query.query.lower()

#         if "how many trees" in user_query:
#             tree_count = predicted_data.get("tree_count", 0)
#             response = f"There are {tree_count} trees detected in the satellite imagery."
#         elif "total area" in user_query:
#             total_area = predicted_data.get("total_area", 0.0)
#             response = f"The total area of the satellite imagery is {total_area} square kilometers."
#         else:
#             client = Mistral(api_key=api_key)
#             model = "mistral-large-latest"
#             chat_response = client.chat.complete(
#                 model=model,
#                 messages=[
#                     {"role": "user", "content": query.query}
#                 ]
#             )
#             response = chat_response.choices[0].message.content
#         return {"query": query.query, "response": response}
#     except Exception as e:
#         logging.error(f"Error in /chat endpoint: {e}")
#         return JSONResponse(status_code=500, content={"message": "An error occurred during chat processing."})

# # Run the application
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000)


import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import torch
from torchvision import transforms
import uvicorn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import Affine
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral
from torchvision.models.detection.transform import GeneralizedRCNNTransform  # Import GeneralizedRCNNTransform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv('MISTRAL_API_KEY')
if api_key is None:
    raise ValueError("Please set the MISTRAL_API_KEY environment variable.")

predicted_data = {
    "tree_count": None,
    "total_area": None,
}

class ChatQuery(BaseModel):
    query: str

app = FastAPI()

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting this in production
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

# Helper function to create shapely box
def box_shape(coords):
    minx, miny, maxx, maxy = coords
    return box(minx, miny, maxx, maxy)

@app.post("/predict")
async def predict(image: UploadFile = File(...), model_name: str = Query(...)):
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

        # Normalize the image array to [0, 255] if not already
        if img_array.dtype != np.uint8:
            img_min = img_array.min()
            img_max = img_array.max()
            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype('uint8')
            logger.info("Normalized image array to uint8.")

        # Convert NumPy array to PIL Image for compatibility with transforms
        img_pil = Image.fromarray(np.transpose(img_array, (1, 2, 0)), 'RGBA')
        logger.info("Converted NumPy array to PIL Image in 'RGBA' mode.")
    except Exception as e:
        logger.error(f"Error processing image array: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to process image data."})

    try:
        # Image preprocessing transforms
        transform = transforms.ToTensor()  # Removed Normalize to prevent conflicts
        logger.info("Defined image preprocessing transforms.")

        # Apply transformations
        input_tensor = transform(img_pil).to(device)
        logger.info(f"Transformed image to tensor with shape {input_tensor.shape}.")

        # Pass as a list to MaskRCNN
        outputs = detection_model([input_tensor])[0]
        logger.info("Performed model inference.")
    except Exception as e:
        logger.error(f"Error during image preprocessing or model inference: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to preprocess image or perform inference."})

    try:
        # Process outputs to get bounding boxes and scores
        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        logger.info(f"Model returned {len(boxes)} boxes with scores.")

        # Filter out low confidence detections
        threshold = 0.7
        selected_indices = scores >= threshold
        boxes = boxes[selected_indices]
        scores = scores[selected_indices]
        logger.info(f"Filtered detections with threshold {threshold}. Remaining boxes: {len(boxes)}.")

        # Update predicted data
        predicted_data['tree_count'] = len(boxes)
        # Calculate total_area based on boxes and image metadata
        predicted_data['total_area'] = calculate_total_area(boxes, transform_raster)  # Implement this function as needed

        # For visualization, ensure the image is in the correct format
        img_rgb = np.array(img_pil.convert('RGB'))  # Convert to RGB for plotting
        logger.info("Converted image to RGB for visualization.")

        # Draw bounding boxes on the image
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(img_rgb)

        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height,
                                 linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # Save the output image
        output_image_path = os.path.join('outputs', f'overlay_{image.filename}.png')
        plt.axis('off')
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)
        logger.info(f"Saved overlay image to '{output_image_path}'.")

        return FileResponse(output_image_path, media_type="image/png")
    except Exception as e:
        logger.error(f"Error processing model outputs: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to process model outputs."})

def calculate_total_area(boxes, transform_raster):
    """
    Placeholder function to calculate total area based on bounding boxes and raster transform.
    You need to implement this based on your specific requirements and raster metadata.
    """
    # Example implementation (assuming boxes are in pixel coordinates):
    total_area = 0.0
    pixel_size_x = transform_raster.a
    pixel_size_y = -transform_raster.e  # Typically negative
    for box_coords in boxes:
        x1, y1, x2, y2 = box_coords
        width = x2 - x1
        height = y2 - y1
        area = width * height * pixel_size_x * pixel_size_y  # Adjust units as needed
        total_area += area
    return total_area

# Endpoint for exporting data
@app.get("/export")
async def export_data(format: str):
    try:
        if format not in ['shp', 'geojson']:
            return JSONResponse(status_code=400, content={"message": "Invalid format. Choose 'shp' or 'geojson'."})
        file_path = os.path.join('exports', f'exported_data.{format}')
        if os.path.exists(file_path):
            media_type = "application/octet-stream"
            return FileResponse(file_path, media_type=media_type, filename=f'exported_data.{format}')
        else:
            return JSONResponse(status_code=404, content={"message": "File not found."})
    except Exception as e:
        logger.error(f"Error in /export endpoint: {e}")
        return JSONResponse(status_code=500, content={"message": "An error occurred during data export."})

@app.post("/chat/")
async def chat_with_model(query: ChatQuery):
    try:
        user_query = query.query.lower()

        if "how many trees" in user_query:
            tree_count = predicted_data.get("tree_count", 0)
            response = f"There are {tree_count} trees detected in the satellite imagery."
        elif "total area" in user_query:
            total_area = predicted_data.get("total_area", 0.0)
            response = f"The total area of the detected regions is {total_area} square units."
        else:
            client = Mistral(api_key=api_key)
            model = "mistral-large-latest"
            chat_response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "user", "content": query.query}
                ]
            )
            response = chat_response.choices[0].message.content
        return {"query": query.query, "response": response}
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        return JSONResponse(status_code=500, content={"message": "An error occurred during chat processing."})

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

