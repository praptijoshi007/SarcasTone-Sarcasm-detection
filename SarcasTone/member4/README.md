# Member 4: Explainability, Robustness, Backend, Frontend & Deployment

Welcome to the completed fourth module of **SarcasTone: Multimodal Sarcasm Detection Using Textual Semantics and Speech Prosody**.

---

## Architecture & Integration Overview

Member 4 unifies the preceding three upstream modules into a full-stack, research-grade production system:
- **Upstream Integration**:
  - `member1_text/`: Text span-pooling encoder, normalization, and ASR transcription.
  - `member2_audio/`: Wav2Vec 2.0 acoustic representations, 32 prosodic descriptors, and 6-dimensional signal quality checks.
  - `member3_fusion/`: Bidirectional stacked cross-attention, Prosodic-Semantic Incongruence Dissonance (PSID), and confidence-gated fusion.
- **Member 4 Modules**:
  - `backend/`: Production FastAPI server supporting file uploads, validation, logging, and REST endpoints.
  - `frontend/`: Interactive Streamlit dashboard with live audio recording, F0 pitch/energy plotting, and XAI token/acoustic drivers.
  - `xai/`: Axiomatic Captum Integrated Gradients, prosodic feature attributions, and predictive entropy calibration.
  - `experiments/`: Robustness suite evaluating 8 real-world noise, degradation, and dropout conditions.
  - `research/`: Academic literature review and complete paper draft with verified DOIs and empirical benchmark tables.
  - `docker/` & `deployment/`: Production Dockerfile, docker-compose, and deployment instructions.

---

## Directory Layout

```
member4/
├── backend/
│   ├── __init__.py
│   ├── main.py            # FastAPI production application
│   ├── pipeline.py        # 4-way MultimodalPipeline orchestrator
│   └── schemas.py         # Pydantic request/response schemas
├── configs/
│   └── system_config.yaml # Central system & interface contract config
├── deployment/
│   └── DEPLOYMENT.md      # Exact commands for local, Docker, & Colab
├── docker/
│   ├── Dockerfile         # Multi-stage production container build
│   └── entrypoint.sh      # Service startup script
├── experiments/
│   ├── __init__.py
│   └── robustness.py      # Automated 8-scenario stress-testing suite
├── fixtures/
│   └── preset_sheldon_sarcastic.wav  # Preset test sample
├── frontend/
│   └── app.py             # Full-featured Streamlit dashboard
├── mlops/
│   ├── __init__.py
│   └── tracking.py        # MLflow experiment tracking & model versioning
├── reports/
│   ├── robustness_report.json # Benchmark results JSON
│   └── robustness_report.md   # Rendered Markdown robustness report
├── research/
│   ├── literature_review.md   # In-depth literature review & verified bibliography
│   └── paper_draft.md         # Full publication-ready research paper
├── tests/
│   ├── __init__.py
│   └── test_api.py        # Comprehensive FastAPI integration tests
├── docker-compose.yml     # Multi-container orchestration
├── README.md              # Module documentation
└── requirements.txt       # Locked dependencies
```

---

## Quickstart Commands

### 1. Verify 4-Way Synchronization
```bash
python verify_all_members_sync.py
```

### 2. Run API Integration Tests
```bash
pytest member4/tests/test_api.py -v
```

### 3. Launch Backend
```bash
uvicorn member4.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Launch Frontend Dashboard
```bash
streamlit run member4/frontend/app.py --server.port 8501
```
