from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import torch
from imageprocess import ImageSegmenter
import os
import tempfile
import imghdr

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the segmenter
segmenter = None

# List of allowed image formats
ALLOWED_IMAGE_TYPES = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

def validate_image(file_content: bytes) -> bool:
    """Validate if the file is actually an image"""
    image_type = imghdr.what(None, file_content)
    return image_type in ALLOWED_IMAGE_TYPES

@app.on_event("startup")
async def startup_event():
    global segmenter
    model_path = os.getenv("MODEL_PATH", "./saved_models/isnet.pth")
    segmenter = ImageSegmenter(model_path=model_path)

@app.get("/")
async def root():
    return {"message": "Image Segmentation API is running"}

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        # Validate if it's actually an image
        if not validate_image(content):
            raise HTTPException(
                status_code=400, 
                detail="Invalid image format. Supported formats: JPG, JPEG, PNG, BMP, WEBP"
            )

        # Create a temporary file with the original extension
        original_extension = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_extension) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Open and convert image to ensure format compatibility
            with Image.open(temp_path) as img:
                # Convert to RGB if image is in RGBA or other formats
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Save as PNG for processing
                png_path = temp_path + '.png'
                img.save(png_path, 'PNG')
            
            # Process the image
            rgba_image, mask = segmenter.process_image(png_path)

        finally:
            # Clean up temporary files
            os.unlink(temp_path)
            if os.path.exists(png_path):
                os.unlink(png_path)

        # Convert RGBA image to bytes
        img_byte_arr = io.BytesIO()
        rgba_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(
            content=img_byte_arr, 
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=segmented_{file.filename.rsplit('.', 1)[0]}.png"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-formats")
async def supported_formats():
    """Return list of supported image formats"""
    return {"supported_formats": list(ALLOWED_IMAGE_TYPES)}

# # For local development
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=80)