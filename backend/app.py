# import os
# import logging
# from fastapi import FastAPI, File, UploadFile, Query
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# import shutil
# import torch
# import math
# from torchvision import transforms
# import uvicorn
# import numpy as np
# from PIL import Image
# from torchvision.models.detection import MaskRCNN
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from torchvision.models.detection.transform import GeneralizedRCNNTransform
# from rasterio.transform import Affine
# import cv2
# import geopandas as gpd
# from shapely.geometry import Polygon

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # CORS settings
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Consider restricting this in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Create necessary directories
# os.makedirs('uploads', exist_ok=True)
# os.makedirs('outputs', exist_ok=True)
# os.makedirs('exports', exist_ok=True)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logger.info(f"Using device: {device}")

# # Directory where your models are stored
# MODEL_DIR = 'models'

# # Cache to store loaded models to avoid reloading
# loaded_models = {}

# def get_detection_model(num_classes, in_channels):
#     try:
#         # Create a ResNet-50 backbone with FPN
#         backbone = resnet_fpn_backbone('resnet50', pretrained=False)

#         # Modify the conv1 layer to accept the required number of input channels
#         backbone.body.conv1 = torch.nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=64,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False
#         )

#         # Create the MaskRCNN model
#         model = MaskRCNN(backbone, num_classes=num_classes)
#         return model
#     except Exception as e:
#         logger.error(f"Error creating detection model: {e}")
#         raise RuntimeError("Failed to create the detection model.")

# def load_model(model_name):
#     if model_name in loaded_models:
#         logger.info(f"Model '{model_name}' loaded from cache.")
#         return loaded_models[model_name]

#     model_path = os.path.join(MODEL_DIR, model_name)
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model '{model_name}' not found in '{MODEL_DIR}' directory.")

#     try:
#         # Load the state_dict
#         state_dict = torch.load(model_path, map_location=device)
#         logger.info(f"Loaded state_dict from '{model_path}'.")

#         # Determine the number of input channels
#         conv1_weights = state_dict['backbone.body.conv1.weight']
#         in_channels = conv1_weights.shape[1]
#         logger.info(f"conv1 weights have {in_channels} channels.")

#     except Exception as e:
#         logger.error(f"Error loading state_dict from '{model_path}': {e}")
#         raise RuntimeError(f"Failed to load state_dict from '{model_path}'.")

#     num_classes = 2  # Adjust as per your model (e.g., 1 class + background)
#     model = get_detection_model(num_classes, in_channels)

#     try:
#         # Load the state_dict
#         model.load_state_dict(state_dict)
#         logger.info(f"Loaded state_dict into model '{model_name}'.")
#     except Exception as e:
#         logger.error(f"Error loading state_dict into model: {e}")
#         raise RuntimeError(f"Failed to load state_dict into model.")

#     # Modify the model's internal transforms to handle the correct number of channels
#     try:
#         if in_channels == 3:
#             image_mean = [0.485, 0.456, 0.406]
#             image_std = [0.229, 0.224, 0.225]
#         elif in_channels == 4:
#             image_mean = [0.485, 0.456, 0.406, 0.5]
#             image_std = [0.229, 0.224, 0.225, 0.25]
#         else:
#             raise ValueError(f"Unsupported number of input channels: {in_channels}")

#         model.transform = GeneralizedRCNNTransform(
#             min_size=800,
#             max_size=1333,
#             image_mean=image_mean,
#             image_std=image_std
#         )
#         logger.info(f"Updated model's internal transforms to handle {in_channels} channels.")
#     except Exception as e:
#         logger.error(f"Error updating model's transforms: {e}")
#         raise RuntimeError("Failed to update model's internal transforms.")

#     model.to(device)
#     model.eval()
#     loaded_models[model_name] = model
#     logger.info(f"Model '{model_name}' loaded and cached successfully.")
#     return model

# @app.get("/models")
# async def list_models():
#     try:
#         models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
#         if not models:
#             return JSONResponse(status_code=404, content={"message": "No models found."})
#         return {"models": models}
#     except Exception as e:
#         logger.error(f"Error listing models: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to list models."})

# @app.post("/predict")
# async def predict(
#     image: UploadFile = File(...),
#     model_name: str = Query(...),
#     threshold: float = Query(0.5, ge=0.0, le=1.0)
# ):
#     try:
#         # Load the selected model
#         detection_model = load_model(model_name)
#     except FileNotFoundError as e:
#         logger.error(f"Model not found: {e}")
#         return JSONResponse(status_code=404, content={"message": str(e)})
#     except Exception as e:
#         logger.error(f"Error loading model: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to load the selected model."})

#     try:
#         # Save uploaded image
#         image_path = os.path.join('uploads', image.filename)
#         with open(image_path, "wb") as buffer:
#             shutil.copyfileobj(image.file, buffer)
#         logger.info(f"Saved uploaded image to '{image_path}'.")
#     except Exception as e:
#         logger.error(f"Error saving uploaded image: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to save uploaded image."})

#     try:
#         # Read the image using PIL
#         img_pil = Image.open(image_path)
#         original_width, original_height = img_pil.size
#         logger.info(f"Image size: {original_width} x {original_height}")

#         # Adjust image channels to match the model's expected input channels
#         in_channels = detection_model.backbone.body.conv1.in_channels
#         logger.info(f"Model expects {in_channels} input channels.")

#         if in_channels == 3:
#             img_pil = img_pil.convert('RGB')
#         elif in_channels == 4:
#             if img_pil.mode == 'RGBA':
#                 img_pil = img_pil
#             else:
#                 # Add an extra channel (e.g., duplicate the red channel)
#                 img_pil = img_pil.convert('RGB')
#                 img_array = np.array(img_pil)
#                 extra_channel = img_array[:, :, 0:1]  # Use the red channel
#                 img_array = np.concatenate((img_array, extra_channel), axis=-1)
#                 img_pil = Image.fromarray(img_array)
#                 logger.info("Added an extra channel to the image to match the model's expected input channels.")
#         else:
#             raise ValueError(f"Unsupported number of input channels: {in_channels}")

#         crs = None
#         transform_raster = Affine.identity()
#         logger.info("Set default CRS and transform.")
#     except Exception as e:
#         logger.error(f"Error preparing image: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to prepare the image for processing."})

#     try:
#         # Image preprocessing transforms
#         transform = transforms.ToTensor()
#         logger.info("Defined image preprocessing transforms.")

#         # Define tile size and overlap
#         tile_size = 1024  # Adjust based on your GPU memory capacity
#         overlap = 100  # Overlap between tiles to avoid edge effects

#         # Calculate number of tiles in each dimension
#         n_tiles_x = math.ceil((original_width - overlap) / (tile_size - overlap))
#         n_tiles_y = math.ceil((original_height - overlap) / (tile_size - overlap))
#         logger.info(f"Tiling image into {n_tiles_x} x {n_tiles_y} tiles.")

#         # Initialize lists to store detections from all tiles
#         all_polygons = []
#         all_areas = []
#         mask_overlay = np.zeros((original_height, original_width), dtype=np.uint8)

#         # Loop over tiles
#         for i in range(n_tiles_x):
#             for j in range(n_tiles_y):
#                 # Calculate tile boundaries with overlap
#                 x_start = i * (tile_size - overlap)
#                 y_start = j * (tile_size - overlap)
#                 x_end = x_start + tile_size
#                 y_end = y_start + tile_size

#                 # Ensure the tile does not exceed image boundaries
#                 x_end = min(x_end, original_width)
#                 y_end = min(y_end, original_height)

#                 # Crop the tile from the image
#                 tile = img_pil.crop((x_start, y_start, x_end, y_end))

#                 # Apply transformations
#                 input_tensor = transform(tile).to(device)
#                 logger.info(f"Processing tile at position ({i}, {j}) with shape {input_tensor.shape}.")

#                 # Perform inference with no gradient calculation
#                 with torch.no_grad():
#                     outputs = detection_model([input_tensor])[0]

#                 # Process outputs to get masks and scores
#                 scores = outputs['scores'].cpu().numpy()
#                 masks = outputs['masks'].cpu().numpy()  # shape: [N, 1, H, W]

#                 # Filter out detections based on the threshold
#                 selected_indices = scores >= threshold
#                 scores = scores[selected_indices]
#                 masks = masks[selected_indices]

#                 for k, mask in enumerate(masks):
#                     # The mask is of shape [1, H, W], we need to convert it to [H, W]
#                     mask = mask[0]
#                     # Threshold the mask at 0.5
#                     mask = (mask >= 0.5).astype(np.uint8)
#                     # Resize mask to tile size if necessary
#                     mask_height, mask_width = mask.shape
#                     if mask.shape != (tile.size[1], tile.size[0]):
#                         mask = cv2.resize(mask, (tile.size[0], tile.size[1]), interpolation=cv2.INTER_NEAREST)

#                     # Place the mask in the correct position in the full-size mask
#                     mask_full = np.zeros((original_height, original_width), dtype=np.uint8)
#                     mask_full[y_start:y_end, x_start:x_end] = mask

#                     # Find contours
#                     contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#                     for contour in contours:
#                         # Convert contour coordinates to x, y lists
#                         contour = contour.reshape(-1, 2)
#                         # Use pixel coordinates directly
#                         spatial_coords = []
#                         for x, y in contour:
#                             spatial_coords.append((x, y))
#                         # Create a polygon
#                         if len(spatial_coords) >= 3:
#                             polygon = Polygon(spatial_coords)
#                             all_polygons.append(polygon)
#                             all_areas.append(polygon.area)
#                     # For visualization, add the mask to the overlay
#                     mask_overlay = cv2.bitwise_or(mask_overlay, mask_full * 255)

#         # Calculate total area
#         total_area = sum(all_areas)
#         tree_count = len(all_polygons)
#         average_tree_area = total_area / tree_count if tree_count > 0 else 0
#         logger.info(f"Calculated total area from masks: {total_area}")

#         # Create a GeoDataFrame with the polygons
#         gdf = gpd.GeoDataFrame(geometry=all_polygons)
#         # Save the GeoDataFrame to a file
#         output_vector_path = os.path.join('outputs', f'detections_{image.filename}.geojson')
#         gdf.to_file(output_vector_path, driver='GeoJSON')
#         logger.info(f"Saved detections to '{output_vector_path}'.")

#         # For visualization, overlay the masks on the image
#         img_rgb = np.array(img_pil.convert('RGB'))  # Ensure image is in RGB
#         # Create a colored mask
#         colored_mask = np.zeros_like(img_rgb)
#         colored_mask[:, :, 1] = mask_overlay  # Highlight in green
#         # Overlay the mask on the image
#         overlayed_image = cv2.addWeighted(img_rgb, 1.0, colored_mask, 0.5, 0)

#         # Save the output image
#         output_image_path = os.path.join('outputs', f'overlay_{image.filename}.png')
#         cv2.imwrite(output_image_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
#         logger.info(f"Saved overlay image to '{output_image_path}'.")

#         # Read image bytes
#         with open(output_image_path, "rb") as image_file:
#             image_bytes = image_file.read()

#         # Return predicted data and image
#         response_data = {
#             "tree_count": tree_count,
#             "total_area": total_area,
#             "average_tree_area": average_tree_area,
#             "image": image_bytes.hex(),  # Convert bytes to hex string
#             "geojson_url": f"/outputs/detections_{image.filename}.geojson",
#             "image_metadata": {
#                 "width": original_width,
#                 "height": original_height,
#                 "crs": str(crs),
#                 "transform": str(transform_raster)
#             }
#         }
#         return JSONResponse(content=response_data)
#     except Exception as e:
#         logger.error(f"Error processing model outputs: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to process model outputs."})

# @app.get("/outputs/{filename}")
# async def get_output_file(filename: str):
#     file_path = os.path.join('outputs', filename)
#     if os.path.exists(file_path):
#         return FileResponse(file_path, media_type='application/octet-stream', filename=filename)
#     else:
#         return JSONResponse(status_code=404, content={"message": "File not found."})

# # Run the application
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000)

import os
import logging
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Added for serving static files
import shutil
import torch
import math
from torchvision import transforms
import uvicorn
import numpy as np
from PIL import Image
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from rasterio.transform import Affine
import cv2
import geopandas as gpd
from shapely.geometry import Polygon

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Mount the outputs directory to serve static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

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

def get_detection_model(num_classes, in_channels):
    try:
        # Create a ResNet-50 backbone with FPN
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)

        # Modify the conv1 layer to accept the required number of input channels
        backbone.body.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
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

def load_model(model_name):
    if model_name in loaded_models:
        logger.info(f"Model '{model_name}' loaded from cache.")
        return loaded_models[model_name]

    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found in '{MODEL_DIR}' directory.")

    try:
        # Load the state_dict
        state_dict = torch.load(model_path, map_location=device)
        logger.info(f"Loaded state_dict from '{model_path}'.")

        # Determine the number of input channels
        conv1_weights = state_dict['backbone.body.conv1.weight']
        in_channels = conv1_weights.shape[1]
        logger.info(f"conv1 weights have {in_channels} channels.")

    except Exception as e:
        logger.error(f"Error loading state_dict from '{model_path}': {e}")
        raise RuntimeError(f"Failed to load state_dict from '{model_path}'.")

    num_classes = 2  # Adjust as per your model (e.g., 1 class + background)
    model = get_detection_model(num_classes, in_channels)

    try:
        # Load the state_dict
        model.load_state_dict(state_dict)
        logger.info(f"Loaded state_dict into model '{model_name}'.")
    except Exception as e:
        logger.error(f"Error loading state_dict into model: {e}")
        raise RuntimeError(f"Failed to load state_dict into model.")

    # Modify the model's internal transforms to handle the correct number of channels
    try:
        if in_channels == 3:
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        elif in_channels == 4:
            image_mean = [0.485, 0.456, 0.406, 0.5]
            image_std = [0.229, 0.224, 0.225, 0.25]
        else:
            raise ValueError(f"Unsupported number of input channels: {in_channels}")

        model.transform = GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=image_mean,
            image_std=image_std
        )
        logger.info(f"Updated model's internal transforms to handle {in_channels} channels.")
    except Exception as e:
        logger.error(f"Error updating model's transforms: {e}")
        raise RuntimeError("Failed to update model's internal transforms.")

    model.to(device)
    model.eval()
    loaded_models[model_name] = model
    logger.info(f"Model '{model_name}' loaded and cached successfully.")
    return model

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
        # Read the image using PIL
        img_pil = Image.open(image_path)
        original_width, original_height = img_pil.size
        logger.info(f"Image size: {original_width} x {original_height}")

        # Adjust image channels to match the model's expected input channels
        in_channels = detection_model.backbone.body.conv1.in_channels
        logger.info(f"Model expects {in_channels} input channels.")

        if in_channels == 3:
            img_pil = img_pil.convert('RGB')
        elif in_channels == 4:
            if img_pil.mode == 'RGBA':
                img_pil = img_pil
            else:
                # Add an extra channel (e.g., duplicate the red channel)
                img_pil = img_pil.convert('RGB')
                img_array = np.array(img_pil)
                extra_channel = img_array[:, :, 0:1]  # Use the red channel
                img_array = np.concatenate((img_array, extra_channel), axis=-1)
                img_pil = Image.fromarray(img_array)
                logger.info("Added an extra channel to the image to match the model's expected input channels.")
        else:
            raise ValueError(f"Unsupported number of input channels: {in_channels}")

        crs = None
        transform_raster = Affine.identity()
        logger.info("Set default CRS and transform.")
    except Exception as e:
        logger.error(f"Error preparing image: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to prepare the image for processing."})

    try:
        # Image preprocessing transforms
        transform = transforms.ToTensor()
        logger.info("Defined image preprocessing transforms.")

        # Define tile size and overlap
        tile_size = 1024  # Adjust based on your GPU memory capacity
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

                # Filter out detections based on the threshold
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
                    if mask.shape != (tile.size[1], tile.size[0]):
                        mask = cv2.resize(mask, (tile.size[0], tile.size[1]), interpolation=cv2.INTER_NEAREST)

                    # Place the mask in the correct position in the full-size mask
                    mask_full = np.zeros((original_height, original_width), dtype=np.uint8)
                    mask_full[y_start:y_end, x_start:x_end] = mask

                    # Find contours
                    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        # Convert contour coordinates to x, y lists
                        contour = contour.reshape(-1, 2)
                        # Use pixel coordinates directly
                        spatial_coords = []
                        for x, y in contour:
                            spatial_coords.append((x, y))
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
        average_tree_area = total_area / tree_count if tree_count > 0 else 0
        logger.info(f"Calculated total area from masks: {total_area}")

        # Create a GeoDataFrame with the polygons
        gdf = gpd.GeoDataFrame(geometry=all_polygons)
        # Save the GeoDataFrame to a file
        output_vector_path = os.path.join('outputs', f'detections_{image.filename}.geojson')
        gdf.to_file(output_vector_path, driver='GeoJSON')
        logger.info(f"Saved detections to '{output_vector_path}'.")

        # For visualization, overlay the masks on the image
        img_rgb = np.array(img_pil.convert('RGB'))  # Ensure image is in RGB
        # Create a colored mask
        colored_mask = np.zeros_like(img_rgb)
        colored_mask[:, :, 1] = mask_overlay  # Highlight in green
        # Overlay the mask on the image
        overlayed_image = cv2.addWeighted(img_rgb, 1.0, colored_mask, 0.5, 0)

        # Save the output image
        output_image_path = os.path.join('outputs', f'overlay_{image.filename}.png')
        cv2.imwrite(output_image_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved overlay image to '{output_image_path}'.")

        # Return predicted data and image URL
        response_data = {
            "tree_count": tree_count,
            "total_area": total_area,
            "average_tree_area": average_tree_area,
            "overlay_image_url": f"/outputs/overlay_{image.filename}.png",
            "geojson_url": f"/outputs/detections_{image.filename}.geojson",
            "image_metadata": {
                "width": original_width,
                "height": original_height,
                "crs": str(crs),
                "transform": str(transform_raster)
            }
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error processing model outputs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": "Failed to process model outputs."})

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
