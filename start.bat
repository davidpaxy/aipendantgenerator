@echo off
REM Avvio server FastAPI con Python module (niente PATH richiesto)
REM (opzionale) attiva venv: .venv\Scripts\activate
python -m uvicorn app:app --reload --port 8000
pause
