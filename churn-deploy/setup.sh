#!/usr/bin/env bash
# One-shot setup for Mac / Linux / Git Bash (Windows).
# Usage:  bash setup.sh
set -e

python3 -m venv .venv
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install awsebcli

python train.py

echo
echo "Done. Next:"
echo "  gunicorn --bind=0.0.0.0:9696 predict:app     # run locally"
echo "  python predict_test.py                        # test it"
echo "  code .                                        # open in VS Code"
