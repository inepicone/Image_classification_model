import time
import redis
import os
import settings
import json
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
import numpy as np


# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host = settings.REDIS_IP,
    port = settings.REDIS_PORT,
    db = settings.REDIS_DB_ID
)

# Load ML model and assign to variable `model`
model = resnet50.ResNet50(include_top=True, weights="imagenet")


def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """

    file_path = os.path.join(settings.UPLOAD_FOLDER, image_name)
    img = image.load_img(file_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = resnet50.preprocess_input(img)
    preds = model.predict(img)
    preds_set = resnet50.decode_predictions(preds, top=1)
    pred_class = preds_set[0][0][1]
    pred_score = round(float(preds_set[0][0][2]), 4)

    return pred_class, pred_score


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Take a new job from Redis
        # Run ML model on the given data
        # Store model prediction in a dict with the following shape:
        # {"prediction": str, "score": float}
        # Store the results on Redis using the original job ID as the key
        # so the API can match the results it gets to the original job
        # sent
        
        _, data = db.brpop(settings.REDIS_QUEUE)
        data = json.loads(data)
        pred_class, pred_score = predict(data["image_name"])
        pred = {"class": pred_class, "score": pred_score}
        db.set(data["id"], json.dumps(pred))

        # Don't forget to sleep for a bit at the end
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
