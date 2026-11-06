# Task runner for Windows PowerShell (wraps the same commands as the Makefile / DVC).
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "pipeline", "features", "train", "serve", "dashboard", "test", "lint", "compose")]
    [string]$Task = "test"
)

switch ($Task) {
    "setup"     { python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt }
    "pipeline"  { .\.venv\Scripts\python.exe -m src.pipeline all }
    "features"  { .\.venv\Scripts\python.exe -m src.pipeline features }
    "train"     { .\.venv\Scripts\python.exe -m src.pipeline train }
    "serve"     { .\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 }
    "dashboard" { .\.venv\Scripts\python.exe -m streamlit run frontend/dashboard.py }
    "test"      { .\.venv\Scripts\python.exe -m pytest --cov=src --cov=api }
    "lint"      { .\.venv\Scripts\python.exe -m ruff check .; .\.venv\Scripts\python.exe -m ruff format --check . }
    "compose"   { docker compose up --build }
}
