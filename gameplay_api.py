from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
from pathlib import Path
import uuid
import shutil
import uvicorn
import io
from PIL import Image
import os

# Import the GameplayAnalysis class from your existing code
from gameplay_analysis import GameplayAnalysis

app = FastAPI(title="Gameplay Analysis API")

# Initialize the GameplayAnalysis instance globally
try:
    analyzer = GameplayAnalysis(
        full_map_path="/Users/akshatvaja/Documents/work/MINED_2025/map/full_map.png",
        yolo_model_path="/Users/akshatvaja/Documents/work/MINED_2025/map/player_tracker.pt"
    )
except Exception as e:
    print(f"Failed to initialize analyzer: {str(e)}")
    raise

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("output_maps")
OUTPUT_DIR.mkdir(exist_ok=True)

def process_image_bytes(image_bytes):
    """Convert image bytes to OpenCV format"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    """
    Process a gameplay frame and return the analyzed map image
    
    Parameters:
    - file: Image file (gameplay frame)
    
    Returns:
    - Processed map image with player positions marked
    """
    try:
        # Read and validate the uploaded image
        contents = await file.read()
        frame = process_image_bytes(contents)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process the frame
        result = analyzer.process_frame(frame)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Frame processing failed")
        
        # Generate unique filename for the result
        output_filename = f"result_{uuid.uuid4()}.jpg"
        output_path = OUTPUT_DIR / output_filename
        
        # Save the result
        cv2.imwrite(str(output_path), result)
        
        # Return the processed image
        return FileResponse(
            str(output_path),
            media_type="image/jpeg",
            filename=output_filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_frame_with_details/")
async def process_frame_with_details(file: UploadFile = File(...)):
    """
    Process a gameplay frame and return both the analyzed map image and player coordinates
    
    Parameters:
    - file: Image file (gameplay frame)
    
    Returns:
    - JSON containing:
        - image_url: URL to the processed image
        - player_positions: List of detected player coordinates
    """
    try:
        # Read and validate the uploaded image
        contents = await file.read()
        frame = process_image_bytes(contents)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract minimap and detect players
        minimap = analyzer.extract_minimap(frame)
        player_centers = analyzer.detect_players(minimap)
        
        # Transform positions if homography is available
        transformed_points = []
        if analyzer.homography_matrix is not None:
            transformed_points = analyzer.transform_points(player_centers)
        else:
            # Compute homography if not available
            if analyzer.compute_homography(minimap):
                transformed_points = analyzer.transform_points(player_centers)
        
        # Create visualization
        result = analyzer.visualize_result(transformed_points)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Visualization failed")
        
        # Generate unique filename
        output_filename = f"result_{uuid.uuid4()}.jpg"
        output_path = OUTPUT_DIR / output_filename
        
        # Save the result
        cv2.imwrite(str(output_path), result)
        
        return JSONResponse({
            "image_url": f"/output_maps/{output_filename}",
            "player_positions": transformed_points
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize necessary resources on startup"""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    # Clean up output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)