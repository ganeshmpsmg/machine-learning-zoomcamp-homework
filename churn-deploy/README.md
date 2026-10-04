# Churn Prediction - Deployment (ML Zoomcamp, Module 5.1 - 5.8)

```
churn-deploy/
├── train.py            # 5.1/5.2  train + save model with pickle
├── predict.py          # 5.3/5.4  Flask web service (served with gunicorn)
├── predict_test.py     # test client (local or AWS)
├── requirements.txt    # 5.5      dependencies
├── Dockerfile          # 5.6/5.7  container
├── .dockerignore
├── setup.sh            # one-shot setup
└── .vscode/            # VS Code settings + run tasks
```

## 1. Open in VS Code

```bash
cd churn-deploy
code .
```
(If `code` is not found: VS Code -> Ctrl+Shift+P -> "Shell Command: Install 'code' command in PATH".)

## 2. Setup + train (5.1, 5.2)

Mac / Linux / Git Bash:
```bash
bash setup.sh
```

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install awsebcli
python train.py
```
This downloads the Telco churn CSV and creates `model_C=1.0.bin`.

## 3. Run locally (5.3, 5.4)

Linux/Mac/WSL: `gunicorn --bind=0.0.0.0:9696 predict:app`
Windows (gunicorn does not run on Windows): `python predict.py`

Then in another terminal: `python predict_test.py`

## 4. Docker (5.6 / 5.7)

```bash
docker build -t churn-prediction .
docker run -it --rm -p 9696:9696 churn-prediction
python predict_test.py
```

## 5. Deploy to AWS Elastic Beanstalk (5.8)

One-time AWS setup:
1. Create an AWS account, then an IAM user with the policy `AdministratorAccess-AWSElasticBeanstalk`.
2. Create an access key for that user.
3. Run `aws configure` (or `eb init` will ask for the keys).

Deploy (run inside this folder, with the venv active):
```bash
eb init -p docker -r ap-south-1 churn-serving
eb local run --port 9696        # optional: test the container locally
eb create churn-serving-env
```
`ap-south-1` is Mumbai; change it if you prefer another region.

When `eb create` finishes it prints the URL (`...ap-south-1.elasticbeanstalk.com`). Test:
```bash
python predict_test.py churn-serving-env.xxxxxxxx.ap-south-1.elasticbeanstalk.com
```

Redeploy after code changes: `eb deploy`

### IMPORTANT - avoid charges
When done, terminate everything:
```bash
eb terminate churn-serving-env
```
The service is public (anyone with the URL can call it), so do not leave it running.

## Test request format

```bash
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d @customer.json
```
Returns: `{"churn": true, "churn_probability": 0.64}`
