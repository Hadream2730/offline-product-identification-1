from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import time

from services.msgit_model_service import describe_product

app = FastAPI(
    title="MSGIT‑base Product Caption API",
    summary="Generate a product name and description fof the product in the image."
)


@app.post("/analyze", summary="Return product name + description")
async def analyze_image(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    print(f"[API] Request received - File: {file.filename}, Type: {file.content_type}, Size: {file.size} bytes")
    
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        print(f"[API] ERROR: Unsupported file type: {file.content_type}")
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    print(f"[API] Reading image bytes...")
    read_start = time.perf_counter()
    image_bytes = await file.read()
    read_time = time.perf_counter() - read_start
    print(f"[API] Image read completed in {read_time:.3f}s - {len(image_bytes)} bytes")
    
    print(f"[API] Starting product analysis...")
    analysis_start = time.perf_counter()
    result = describe_product(image_bytes)
    analysis_time = time.perf_counter() - analysis_start
    print(f"[API] Analysis completed in {analysis_time:.3f}s")
    
    total_time = time.perf_counter() - start_time
    print(f"[API] Total request time: {total_time:.3f}s")
    print(f"[API] Result: {result}")
    
    return JSONResponse(content=result)

