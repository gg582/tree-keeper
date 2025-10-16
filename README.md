# DoDAG Language Model Prototype

This repository contains a research prototype implementing a language model
that learns over a directed acyclic graph (DoDAG) derived from token
transitions rather than conventional sequential transformer layers. Each node
represents a vocabulary item and edges capture parent→child dependencies.

## Features

- Python training stack with PyTorch implementing parent→child bilinear scoring
  and negative sampling.
- Optional CUDA/C++ extension used to accelerate bilinear scoring when NVCC is
  available; a pure PyTorch fallback is provided.
- Utilities to build DoDAG graphs from Wikipedia-derived corpora or sample
  texts.
- FastAPI backend exposing inference on TCP port `11435`.

## Repository Layout

```
.
├── backend/                 # FastAPI prediction service
├── data/                    # Example corpora
├── dodag_lm/                # Core library modules
├── scripts/                 # Utility scripts (e.g. Wikipedia downloader)
└── train.py                 # CLI entry point for training
```

## Getting Started

1. **Install dependencies** (Python 3.10+ recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare data**. A small toy corpus is provided at
   `data/sample_corpus.txt`. To fetch a lightweight Wikipedia sample via the
   `wikipedia` Python package you can run:

   ```bash
   python scripts/prepare_sample_wiki.py "Large language model" --sentences 50 --output data/wiki_sample.txt
   ```

3. **Train the model** using the CLI:

   ```bash
   python train.py data/sample_corpus.txt --epochs 2 --embedding-dim 64 --hidden-dim 96
   ```

   The command saves a `model.pt` checkpoint containing the model weights,
   vocabulary, and configuration. Training automatically utilises CUDA if
   available but also runs on CPU for experimentation.

4. **Serve predictions** via the FastAPI backend:

   ```bash
   export DODAG_MODEL_PATH=model.pt
   uvicorn backend.server:app --host 0.0.0.0 --port 11435
   ```

   or run `python backend/server.py` which launches the bundled server helper.

   Example request:

   ```bash
   curl -X POST http://localhost:11435/predict \
        -H "Content-Type: application/json" \
        -d '{"context": ["language"], "top_k": 3}'
   ```

5. **Chat interactively** with a trained checkpoint using the bundled CLI:

   ```bash
   python scripts/chatbot.py model.pt --max-tokens 20 --top-k 5 --temperature 0.9
   ```

## CUDA Extension

The CUDA implementation located in `dodag_lm/csrc` accelerates the bilinear
scoring kernel. The code automatically falls back to a PyTorch implementation if
NVCC is unavailable or compilation fails, ensuring compatibility with CPU-only
setups or constrained GPUs such as the RTX 3070 (8 GB VRAM).

## Notes

- The repository ships without a full Wikipedia dataset. Use the helper script
  or other tooling to generate manageable corpora for experimentation before
  scaling to larger pretraining runs.
- The design emphasises modularity so that additional training heuristics (e.g.,
  curriculum learning over graph depth) can be plugged into the existing
  training loop.
