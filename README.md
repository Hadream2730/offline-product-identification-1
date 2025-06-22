# Product Identification API

A FastAPI service that identifies products from images using Microsoft's GIT-base model.

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hadream2730/offline-product-identification-1.git
   cd offline-product-identification-1
   ```

2. **Create virtual environment**
   ```bash
   python -m venv gitenv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   gitenv\Scripts\activate
   
   # macOS/Linux
   source gitenv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

1. Open your browser to `http://localhost:8000/docs`
2. Click on the `/analyze` endpoint
3. Upload an image file (JPEG, PNG, or WebP)
4. Get product name and description in JSON format

## Example Response

```json
{
    "filename": "product.jpg",
    "name": "Running Shoes",
    "description": "A pair of running shoes on a white background."
}
```

## Troubleshooting

- **spaCy model error**: Run `python -m spacy download en_core_web_sm`
- **Out of memory**: The system will automatically use CPU instead of GPU 
