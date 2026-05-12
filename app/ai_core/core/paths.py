from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
AUDIO_DIR = ARTIFACTS_DIR / 'audio'
VECTORSTORE_DIR = ARTIFACTS_DIR / 'vectorstore'
