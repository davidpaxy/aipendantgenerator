#!/bin/bash
# Avvio server FastAPI con Python module (niente PATH richiesto)
# (opzionale) attiva venv: source .venv/bin/activate
python -m uvicorn app:app --reload --port 8000
