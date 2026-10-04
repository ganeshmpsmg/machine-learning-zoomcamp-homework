import json

import onnxruntime as ort
from keras_image_helper import create_preprocessor

CLASSES = [
    "dress", "hat", "longsleeve", "outwear", "pants",
    "shirt", "shoes", "shorts", "skirt", "t-shirt",
]

# Loaded once per container (cold start), reused for every request (warm start)
preprocessor = create_preprocessor("xception", target_size=(299, 299))
session = ort.InferenceSession(
    "clothing-model-new.onnx", providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def predict(url):
    X = preprocessor.from_url(url)
    scores = session.run([output_name], {input_name: X})[0][0].tolist()
    result = dict(zip(CLASSES, scores))
    best = max(result, key=result.get)
    return {"prediction": best, "scores": result}


def response(code, body):
    return {
        "statusCode": code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    try:
        # API Gateway sends the JSON as a string in event["body"];
        # a direct invoke / local test sends the dict itself.
        data = json.loads(event["body"]) if "body" in event else event
        url = data.get("url")
        if not url:
            return response(400, {"error": "Missing 'url' in request body"})
        return response(200, predict(url))
    except json.JSONDecodeError:
        return response(400, {"error": "Body is not valid JSON"})
    except Exception as e:
        return response(500, {"error": f"Prediction failed: {e}"})
