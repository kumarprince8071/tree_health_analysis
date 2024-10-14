# 1st approach
# @app.post("/predict")
# async def predict(
#     image: UploadFile = File(...),
#     model_name: str = Query(...),
#     threshold: float = Query(0.5, ge=0.0, le=1.0)  # Added threshold parameter with default value
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
#         # Read the image using Rasterio
#         with rasterio.open(image_path) as src:
#             img_array = src.read()  # Read all bands
#             crs = src.crs
#             transform_raster = src.transform
#             original_width = src.width
#             original_height = src.height
#             logger.info(f"Image CRS: {crs}")
#             logger.info(f"Image transform: {transform_raster}")
#             logger.info(f"Original image size: {original_width} x {original_height}")
#         logger.info(f"Read image '{image_path}' with Rasterio.")
#     except Exception as e:
#         logger.error(f"Error reading image with Rasterio: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to read image file."})

#     try:
#         # Ensure the image has exactly 4 bands
#         if img_array.shape[0] == 4:
#             logger.info("Image has 4 bands.")
#         elif img_array.shape[0] > 4:
#             img_array = img_array[:4, :, :]  # Take the first 4 bands
#             logger.info("Image has more than 4 bands. Truncated to first 4 bands.")
#         else:
#             # Handle images with fewer than 4 bands by duplicating channels
#             channels_needed = 4 - img_array.shape[0]
#             extra_channels = np.repeat(img_array[0:1, :, :], channels_needed, axis=0)
#             img_array = np.concatenate((img_array, extra_channels), axis=0)
#             logger.info(f"Image had fewer than 4 bands. Duplicated first band to make 4 bands.")

#         # Convert NumPy array to PIL Image for compatibility with transforms
#         # Transpose the array to (H, W, C)
#         img_array_transposed = np.transpose(img_array, (1, 2, 0))
#         # Convert to uint8 if necessary
#         if img_array_transposed.dtype != np.uint8:
#             img_min = img_array_transposed.min()
#             img_max = img_array_transposed.max()
#             img_array_transposed = ((img_array_transposed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
#             logger.info("Normalized image array to uint8.")
#         img_pil = Image.fromarray(img_array_transposed)
#         logger.info("Converted NumPy array to PIL Image.")

#     except Exception as e:
#         logger.error(f"Error processing image array: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to process image data."})

#     try:
#         # Image preprocessing transforms
#         transform = transforms.ToTensor()
#         logger.info("Defined image preprocessing transforms.")

#         # Apply transformations
#         input_tensor = transform(img_pil).to(device)
#         logger.info(f"Transformed image to tensor with shape {input_tensor.shape}.")

#         # Ensure the model is in evaluation mode
#         detection_model.eval()

#         # Perform inference with no gradient calculation
#         with torch.no_grad():
#             outputs = detection_model([input_tensor])[0]
#         logger.info("Performed model inference.")
#     except Exception as e:
#         logger.error(f"Error during image preprocessing or model inference: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to preprocess image or perform inference."})

#     try:
#         # Process outputs to get masks and scores
#         scores = outputs['scores'].cpu().numpy()
#         masks = outputs['masks'].cpu().numpy()  # shape: [N, 1, H, W]
#         logger.info(f"Model returned {len(masks)} masks with scores.")

#         # Validate the threshold value
#         if not (0.0 <= threshold <= 1.0):
#             threshold = 0.5  # Reset to default if invalid
#             logger.warning(f"Invalid threshold received. Reset to default value {threshold}.")

#         # Filter out detections based on the dynamic threshold
#         selected_indices = scores >= threshold
#         scores = scores[selected_indices]
#         masks = masks[selected_indices]
#         logger.info(f"Filtered detections with threshold {threshold}. Remaining masks: {len(masks)}.")

#         # Initialize lists to store spatial polygons and areas
#         polygons = []
#         areas = []
#         # For visualization, create an overlay image
#         mask_overlay = np.zeros((original_height, original_width), dtype=np.uint8)

#         for i, mask in enumerate(masks):
#             # The mask is of shape [1, H, W], we need to convert it to [H, W]
#             mask = mask[0]
#             # Threshold the mask at 0.5
#             mask = (mask >= 0.5).astype(np.uint8)
#             # Resize mask to original image size if necessary
#             if mask.shape != (original_height, original_width):
#                 mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

#             # Find contours
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             for contour in contours:
#                 # Convert contour coordinates to x, y lists
#                 contour = contour.reshape(-1, 2)
#                 # Map pixel coordinates to spatial coordinates
#                 spatial_coords = []
#                 for x, y in contour:
#                     row, col = y, x
#                     lon, lat = rasterio.transform.xy(transform_raster, row, col)
#                     spatial_coords.append((lon, lat))
#                 # Create a polygon
#                 if len(spatial_coords) >= 3:
#                     polygon = Polygon(spatial_coords)
#                     polygons.append(polygon)
#                     areas.append(polygon.area)
#             # For visualization, add the mask to the overlay
#             mask_overlay = cv2.bitwise_or(mask_overlay, mask * 255)

#         # Calculate total area
#         total_area = sum(areas)
#         tree_count = len(polygons)
#         logger.info(f"Calculated total area from masks: {total_area}")

#         # Create a GeoDataFrame with the polygons
#         gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
#         # Save the GeoDataFrame to a file
#         output_vector_path = os.path.join('outputs', f'detections_{image.filename}.geojson')
#         gdf.to_file(output_vector_path, driver='GeoJSON')
#         logger.info(f"Saved detections to '{output_vector_path}'.")

#         # For visualization, overlay the masks on the image
#         img_rgb = np.array(img_pil.convert('RGB'))  # Convert to RGB
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
#             "image": image_bytes.hex(),  # Convert bytes to hex string
#             "geojson_url": f"/outputs/detections_{image.filename}.geojson"  # Include geojson URL if needed
#         }
#         return JSONResponse(content=response_data)
#     except Exception as e:
#         logger.error(f"Error processing model outputs: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to process model outputs."})

# @app.post("/predict")
# async def predict(image: UploadFile = File(...), model_name: str = Query(...)):
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
#         # Read the image using Rasterio
#         with rasterio.open(image_path) as src:
#             img_array = src.read()  # Read all bands
#             crs = src.crs
#             transform_raster = src.transform
#             logger.info(f"Image CRS: {crs}")
#         logger.info(f"Read image '{image_path}' with Rasterio.")
#     except Exception as e:
#         logger.error(f"Error reading image with Rasterio: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to read image file."})

#     try:
#         # Ensure the image has exactly 4 bands
#         if img_array.shape[0] == 4:
#             logger.info("Image has 4 bands.")
#         elif img_array.shape[0] > 4:
#             img_array = img_array[:4, :, :]  # Take the first 4 bands
#             logger.info("Image has more than 4 bands. Truncated to first 4 bands.")
#         else:
#             # Handle images with fewer than 4 bands by duplicating channels
#             channels_needed = 4 - img_array.shape[0]
#             extra_channels = np.repeat(img_array[0:1, :, :], channels_needed, axis=0)
#             img_array = np.concatenate((img_array, extra_channels), axis=0)
#             logger.info(f"Image had fewer than 4 bands. Duplicated first band to make 4 bands.")

#         # Normalize the image array to [0, 255] if not already
#         if img_array.dtype != np.uint8:
#             img_min = img_array.min()
#             img_max = img_array.max()
#             img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype('uint8')
#             logger.info("Normalized image array to uint8.")

#         # Convert NumPy array to PIL Image for compatibility with transforms
#         img_pil = Image.fromarray(np.transpose(img_array, (1, 2, 0)), 'RGBA')
#         logger.info("Converted NumPy array to PIL Image in 'RGBA' mode.")

#         # Resize the image to a manageable size
#         max_dimension = 1024  # Adjust as needed
#         img_pil.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
#         logger.info(f"Resized image to {img_pil.size}.")
#     except Exception as e:
#         logger.error(f"Error processing image array: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to process image data."})

#     try:
#         # Image preprocessing transforms
#         transform = transforms.ToTensor()
#         logger.info("Defined image preprocessing transforms.")

#         # Apply transformations
#         input_tensor = transform(img_pil).to(device)
#         logger.info(f"Transformed image to tensor with shape {input_tensor.shape}.")

#         # Ensure the model is in evaluation mode
#         detection_model.eval()

#         # Perform inference with no gradient calculation
#         with torch.no_grad():
#             outputs = detection_model([input_tensor])[0]
#         logger.info("Performed model inference.")
#     except Exception as e:
#         logger.error(f"Error during image preprocessing or model inference: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to preprocess image or perform inference."})

#     try:
#         # Process outputs to get masks and scores
#         scores = outputs['scores'].cpu().numpy()
#         masks = outputs['masks'].cpu().numpy()  # shape: [N, 1, H, W]
#         logger.info(f"Model returned {len(masks)} masks with scores.")

#         # Filter out low confidence detections
#         threshold = 0.2
#         selected_indices = scores >= threshold
#         scores = scores[selected_indices]
#         masks = masks[selected_indices]
#         logger.info(f"Filtered detections with threshold {threshold}. Remaining masks: {len(masks)}.")

#         # Initialize lists to store spatial polygons and areas
#         polygons = []
#         areas = []
#         # For visualization, create an overlay image
#         mask_overlay = np.zeros((img_pil.height, img_pil.width), dtype=np.uint8)

#         for i, mask in enumerate(masks):
#             # The mask is of shape [1, H, W], we need to convert it to [H, W]
#             mask = mask[0]
#             # Resize mask to original image size if necessary
#             mask = Image.fromarray(mask).resize(img_pil.size, resample=Image.NEAREST)
#             mask = np.array(mask)
#             # Threshold the mask at 0.5
#             mask = (mask >= 0.5).astype(np.uint8)

#             # Find contours
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             for contour in contours:
#                 # Convert contour coordinates to x, y lists
#                 contour = contour.reshape(-1, 2)
#                 # Map pixel coordinates to spatial coordinates
#                 spatial_coords = []
#                 for x, y in contour:
#                     row, col = y, x
#                     lon, lat = rasterio.transform.xy(transform_raster, row, col)
#                     spatial_coords.append((lon, lat))
#                 # Create a polygon
#                 if len(spatial_coords) >= 3:
#                     polygon = Polygon(spatial_coords)
#                     polygons.append(polygon)
#                     areas.append(polygon.area)
#             # For visualization, add the mask to the overlay
#             mask_overlay = cv2.bitwise_or(mask_overlay, mask * 255)

#         # Calculate total area
#         total_area = sum(areas)
#         tree_count = len(polygons)
#         logger.info(f"Calculated total area from masks: {total_area}")

#         # Create a GeoDataFrame with the polygons
#         gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
#         # Save the GeoDataFrame to a file
#         output_vector_path = os.path.join('outputs', f'detections_{image.filename}.geojson')
#         gdf.to_file(output_vector_path, driver='GeoJSON')
#         logger.info(f"Saved detections to '{output_vector_path}'.")

#         # For visualization, overlay the masks on the image
#         img_rgb = np.array(img_pil.convert('RGB'))  # Convert to RGB
#         mask_rgb = cv2.applyColorMap(mask_overlay, cv2.COLORMAP_JET)
#         overlayed_image = cv2.addWeighted(img_rgb, 0.7, mask_rgb, 0.3, 0)

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
#             "image": image_bytes.hex()  # Convert bytes to hex string
#         }
#         return JSONResponse(content=response_data)
#     except Exception as e:
#         logger.error(f"Error processing model outputs: {e}")
#         return JSONResponse(status_code=500, content={"message": "Failed to process model outputs."})

def calculate_total_area(boxes, transform_raster):
    """
    Calculate total area based on bounding boxes and raster transform.
    """
    # This function is no longer used since we are calculating area from masks
    pass

# @app.post("/chat/")
# async def chat_with_model(query: ChatQuery):
#     try:
#         user_query = query.query.lower()
#         response = None

#         # Custom query handling
#         if re.search(r"(how many|number of|total|overall|)\s+(trees|objects)", user_query):
#             tree_count = query.tree_count
#             response = f"There are {tree_count} trees detected in the satellite imagery."

#         elif re.search(r"(total|overall)\s+area", user_query):
#             total_area = query.total_area
#             response = f"The total area of the detected regions is {total_area:.2f} square units."

#         elif re.search(r"(average)\s+(area|size)", user_query):
#             if query.tree_count > 0:
#                 average_area = query.total_area / query.tree_count
#                 response = f"The average area per tree is {average_area:.2f} square units."
#             else:
#                 response = "No trees detected to calculate the average area."

#         elif re.search(r"(show|display)\s+results", user_query):
#             response = "You can view the detection results in the image displayed on the interface."

#         elif re.search(r"(export|download)\s+data", user_query):
#             response = "You can export the detection data using the export options provided."

#         else:
#             # Use Mistral AI for general queries
#             if not api_key:
#                 response = "Chat functionality is not available because the API key is not set."
#             else:
#                 client = Mistral(api_key=api_key)
#                 model = "mistral-large-latest"
#                 logger.info(f"Sending request to Mistral AI with model: {model}")
#                 chat_response = client.chat.complete(
#                     model=model,
#                     messages=[
#                         {"role": "user", "content": query.query}
#                     ]
#                 )
#                 response = chat_response.choices[0].message.content

#         if not response:
#             response = "I'm sorry, I didn't understand your question. Please try rephrasing it."

#         return {"query": query.query, "response": response}
#     except Exception as e:
#         logger.error(f"Error in /chat endpoint: {e}")
#         return JSONResponse(status_code=500, content={"message": "An error occurred during chat processing."})
