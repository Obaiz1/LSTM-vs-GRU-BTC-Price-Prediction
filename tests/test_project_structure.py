"""
Structure tests — verify all required final-project files exist.
These are lightweight and run in CI (no TensorFlow needed).
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_FILES = [
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "requirements-ci.txt",
    "kubernetes/deployment.yaml",
    "kubernetes/service.yaml",
    "deployment/api.py",
    "src/preprocessing.py",
    "src/prediction_utils.py",
    "training/train_v1_lstm.py",
    "training/train_v2_gru.py",
    ".github/workflows/ci.yml",
    "docs/model_comparison.md",
    "docs/commands_for_screenshots.md",
    "docs/demo_video_flow.md",
]


def test_required_files_exist():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(ROOT, f))]
    assert not missing, f"Missing required files: {missing}"


def test_existing_app_preserved():
    # The original Streamlit app and models must NOT have been removed.
    for f in ["app.py", "data_pipeline.py", "models.py"]:
        assert os.path.exists(os.path.join(ROOT, f)), f"Original file {f} was removed!"
