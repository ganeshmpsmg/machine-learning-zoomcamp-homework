# Serverless Clothing Classifier (ML Zoomcamp 2026, Module 9)

A clothing image classifier (10 classes) served as a serverless API with
**AWS Lambda (container image) + API Gateway**. The Keras model from Module 8
is converted to **ONNX** and run with **ONNX Runtime**.

Classes: dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt, t-shirt

## Structure

    clothing-lambda/
    |-- lambda_function.py          # handler + inference
    |-- clothing-model-new.onnx     # model (download, see below)
    |-- requirements.txt            # onnxruntime, keras-image-helper
    |-- Dockerfile                  # Lambda Python 3.13 base image
    |-- test.py                     # works for local and AWS URL
    `-- README.md

## Get the model

    wget https://github.com/DataTalksClub/machine-learning-zoomcamp/releases/download/dl-models/clothing-model-new.onnx

(To convert your own Keras model, see the official workshop: `09-serverless/workshop`.)

## Run locally

    docker build -t clothing-lambda .
    docker run -it --rm -p 8080:8080 clothing-lambda
    python test.py

## Deploy

    aws ecr create-repository --repository-name clothing-lambda --region ap-south-1
    ECR_URL="<ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com"
    aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin $ECR_URL
    docker tag clothing-lambda $ECR_URL/clothing-lambda:v1
    docker push $ECR_URL/clothing-lambda:v1

Then in the AWS console: Lambda -> Create function -> Container image
(memory 1024 MB, timeout 30 s) -> API Gateway -> REST API -> resource `predict`
-> POST -> Lambda proxy integration -> Deploy to stage `test`.

## Call the API

    curl -X POST "https://<API_ID>.execute-api.ap-south-1.amazonaws.com/test/predict" \
      -H "Content-Type: application/json" \
      -d '{"url": "http://bit.ly/mlbookcamp-pants"}'

Response:

    {"prediction": "pants", "scores": {"dress": -1.9, "...": "...", "pants": 20.1}}

Scores are raw model outputs (logits); the highest one is the prediction.
Errors return JSON: `400` (missing/invalid input) or `500` (prediction failed).

## Clean up (avoid charges)

Delete the API Gateway API, the Lambda function and the ECR repository when done.

## Note

The API is open to everyone. Fine for learning; add auth before real use.
