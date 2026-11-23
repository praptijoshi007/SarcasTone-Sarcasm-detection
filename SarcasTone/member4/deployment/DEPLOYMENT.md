# SarcasTone Deployment & Operations Manual

This guide provides exact, verified commands for running, testing, building, and deploying the complete four-member **SarcasTone** system.

---

## 1. Local Environment Setup

### 1.1 Python Virtual Environment
Requires **Python 3.11+**. From the `SarcasTone` directory:

```bash
# Navigate to the project root
cd SarcasTone

# Create a clean virtual environment
python3.11 -m venv .venv

# Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

# Install core and member requirements
pip install --upgrade pip
pip install -r requirements.txt
pip install -r member4/requirements.txt
```

---

## 2. 4-Way Synchronization Verification

Before running services, verify interface alignment and integration contracts across all 4 modules:

```bash
python verify_all_members_sync.py
```
Expected output:
```
======================================================================
 OVERALL STATUS: ALL 4 MEMBERS ARE FULLY SYNCHRONIZED & OPERATIONAL! [PASSED]
======================================================================
```

To run all automated API and integration tests:
```bash
pytest member4/tests/test_api.py -v
```

---

## 3. Running Services Locally

### 3.1 Start the FastAPI Backend
```bash
uvicorn member4.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Alternative ReDoc Documentation: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health Check Endpoint: `curl http://localhost:8000/health`

### 3.2 Start the Streamlit Frontend Dashboard
In a separate terminal (with the virtualenv activated):
```bash
streamlit run member4/frontend/app.py --server.port 8501
```
- Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 4. Example API Invocations (cURL)

### 4.1 Health Diagnostics
```bash
curl -X GET "http://localhost:8000/health"
```

### 4.2 Text-Only Prediction
```bash
curl -X POST "http://localhost:8000/predict/text" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "It is just a privilege to watch your mind at work.",
       "context": ["I think I solved the puzzle."],
       "speaker": "SHELDON"
     }'
```

### 4.3 Multimodal Prediction (Audio + Text)
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "audio=@member4/fixtures/preset_sheldon_sarcastic.wav;type=audio/wav" \
     -F "text=It is just a privilege to watch your mind at work." \
     -F "context=[\"I think I solved the puzzle.\"]" \
     -F "speaker=SHELDON"
```

---

## 5. Docker Build & Container Deployment

### 5.1 Standalone Docker Build
```bash
# Build production Docker image
docker build -f member4/docker/Dockerfile -t sarcastone:latest .

# Run container exposing both API (8000) and UI (8501)
docker run -d --name sarcastone_app -p 8000:8000 -p 8501:8501 sarcastone:latest
```

### 5.2 Docker Compose Multi-Service Deployment
```bash
cd member4
docker-compose up --build -d
```
To check container logs:
```bash
docker-compose logs -f
```
To shut down containers:
```bash
docker-compose down
```

---

## 6. Google Colab Training Workflow

For GPU-accelerated training on Google Colab:

```python
# 1. Clone repository in Colab cell
!git clone https://github.com/praptijoshi007/SarcasTone-Sarcasm-detection.git
%cd SarcasTone-Sarcasm-detection/SarcasTone

# 2. Install dependencies
!pip install -r requirements.txt
!pip install -r member4/requirements.txt

# 3. Train full cross-attention fusion model with PSID
!python member3_fusion/train_fusion.py --experiment full_model --epochs 30 --batch_size 16

# 4. Run cross-modal evaluation and temperature scaling calibration
!python member3_fusion/evaluate_fusion.py --experiment full_model

# 5. Run robustness stress-testing suite
!python member4/experiments/robustness.py
```

---

## 7. Production CPU Inference Notes

1. **Quantization & Torch Threads**: Set `torch.set_num_threads(4)` in low-core cloud environments to optimize latency.
2. **Audio Preprocessing**: Pre-convert user audio to 16 kHz mono PCM WAV client-side or during API ingest to minimize Librosa resampling overhead.
3. **Model Caching**: The `MultimodalPipeline` leverages persistent memory caching; the first inference run loads PyTorch weights into memory (~200ms), and subsequent inferences run in ~20-40ms on standard x86/ARM CPUs.
