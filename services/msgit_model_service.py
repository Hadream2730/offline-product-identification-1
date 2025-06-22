from pathlib import Path
from typing import Union
import time

import torch
from PIL import Image
import spacy
from transformers import AutoProcessor as GitProcessor, AutoModelForCausalLM as GitForCausalLM


# ─── Constants ────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "microsoft/git-base"
RESIZE_TO = 512
MAX_TOK   = 80

# ─── Heavy‑weight objects (loaded once) ───────────────────────────────────────
print(f"[MODEL] Initializing models on device: {DEVICE}")
print(f"[MODEL] Loading spaCy model...")
start_time = time.perf_counter()
nlp = spacy.load("en_core_web_sm", disable=["ner"])
spacy_time = time.perf_counter() - start_time
print(f"[MODEL] spaCy loaded in {spacy_time:.3f}s")

print(f"[MODEL] Loading GIT processor...")
start_time = time.perf_counter()
processor = GitProcessor.from_pretrained(MODEL_ID)
processor_time = time.perf_counter() - start_time
print(f"[MODEL] GIT processor loaded in {processor_time:.3f}s")

print(f"[MODEL] Loading GIT model...")
start_time = time.perf_counter()
model = GitForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(DEVICE).eval()
model_time = time.perf_counter() - start_time
print(f"[MODEL] GIT model loaded in {model_time:.3f}s")

total_init_time = spacy_time + processor_time + model_time
print(f"[MODEL] All models initialized in {total_init_time:.3f}s")


# ─── Internal helpers ────────────────────────────────────────────────────────
def _generate_caption(img: Image.Image) -> str:
    """Run Microsoft GIT‑base and return a raw caption."""
    print(f"[CAPTION] Processing image with GIT model...")
    
    preprocess_start = time.perf_counter()
    pix = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    preprocess_time = time.perf_counter() - preprocess_start
    print(f"[CAPTION] Image preprocessing: {preprocess_time:.3f}s")
    
    inference_start = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(
            pixel_values=pix,
            max_length=MAX_TOK,
            do_sample=True,
            top_p=0.9,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    inference_time = time.perf_counter() - inference_start
    print(f"[CAPTION] Model inference: {inference_time:.3f}s")
    
    decode_start = time.perf_counter()
    caption = processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    decode_time = time.perf_counter() - decode_start
    print(f"[CAPTION] Token decoding: {decode_time:.3f}s")
    print(f"[CAPTION] Generated: '{caption}'")
    
    return caption


# ─── Public API ───────────────────────────────────────────────────────────────
def describe_product(image: Union[str, Path, bytes]) -> dict:
    total_start = time.perf_counter()
    print(f"\n[PRODUCT] Starting product analysis...")
    
    # Load the image
    print(f"[PRODUCT] Loading image...")
    load_start = time.perf_counter()
    if isinstance(image, (str, Path)):
        path = Path(image)
        img = Image.open(path).convert("RGB")
        filename = path.name
        print(f"[PRODUCT] Loaded from file: {filename}")
    else:
        from io import BytesIO
        img = Image.open(BytesIO(image)).convert("RGB")
        filename = "upload"
        print(f"[PRODUCT] Loaded from bytes: {len(image)} bytes")
    
    original_size = img.size
    print(f"[PRODUCT] Original size: {original_size[0]}x{original_size[1]}")
    
    img.thumbnail((RESIZE_TO, RESIZE_TO))
    new_size = img.size
    load_time = time.perf_counter() - load_start
    print(f"[PRODUCT] Resized to: {new_size[0]}x{new_size[1]} in {load_time:.3f}s")

    # Caption with GIT‑base
    caption_start = time.perf_counter()
    caption = _generate_caption(img)
    caption_time = time.perf_counter() - caption_start
    print(f"[PRODUCT] Caption generation: {caption_time:.3f}s")
    
    # Process with spaCy
    print(f"[PRODUCT] Processing text with spaCy...")
    spacy_start = time.perf_counter()
    doc = nlp(caption)
    spacy_time = time.perf_counter() - spacy_start
    print(f"[PRODUCT] spaCy processing: {spacy_time:.3f}s")

    # First meaningful noun phrase → product name
    print(f"[PRODUCT] Extracting product name...")
    name_start = time.perf_counter()
    name = None
    noun_chunks = list(doc.noun_chunks)
    print(f"[PRODUCT] Found {len(noun_chunks)} noun chunks: {[np.text for np in noun_chunks]}")
    
    for np in doc.noun_chunks:
        words = [t.text for t in np if t.pos_ != "DET"]
        if words:
            name = " ".join(words[:3])
            print(f"[PRODUCT] Selected name from noun chunk: '{name}'")
            break

    if not name:
        nouns = [t.text for t in doc if t.pos_ == "NOUN"]
        name = nouns[0] if nouns else caption.split()[0]
        print(f"[PRODUCT] Fallback name selection: '{name}' (from {len(nouns)} nouns)")
    
    name_time = time.perf_counter() - name_start
    print(f"[PRODUCT] Name extraction: {name_time:.3f}s")
    
    # Build result
    result = {
        "name": name.capitalize(),
        "description": caption.rstrip(" .") + ".",
    }
    
    total_time = time.perf_counter() - total_start
    print(f"[PRODUCT] Total processing time: {total_time:.3f}s")
    print(f"[PRODUCT] Final result: {result}")
    
    return result