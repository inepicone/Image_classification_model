import time
import redis
import settings
import json

from tensorflow.keras.applications  import resnet50
from tensorflow.keras.preprocessing import image

import numpy as np

# In this stage the model catch the information from redis and pre and process it.

# Connect to Redis and assign to variable `db`
db = redis.Redis(       
    host = settings.REDIS_IP, 
    port = settings.REDIS_PORT, 
    db   = settings.REDIS_DB_ID,
)

assert db.ping, "I couldnt connect"

# Load your ML model and assign to variable `model`

# in this case we are gonna use an already trained model called ResNet50 from tensorflow/keras
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
    # This model its train with 224x224 images thats why we change the size:
    
    img = image.load_img(settings.UPLOAD_FOLDER+image_name, target_size=(224, 224))
    
    # we need to create a new dimension:
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # prepro:
    x = resnet50.preprocess_input(x)

    # Get predictions
    preds = model.predict(x)

    # The model its trin to return the probability between 1000 classes
    # with top = 1 we choose the one with the hightes prob. 
    preds       = resnet50.decode_predictions(preds, top=1)
    prediction  = preds[0][0][1]
    score       = preds[0][0][2]

    return prediction.capitalize(), round(float(score),4)


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
        # Take a new job from Redis
        _, data_json       =  db.brpop(settings.REDIS_QUEUE) 
        
        # Lets build a dictionary with the information we got from redis:
        data_dictionary    = json.loads(data_json) 

        #   2. Run your ML model on the given data
        prediction, score     = predict(data_dictionary['image_name'])

        prediction_dictionary = {"prediction": prediction , "score": score }
        
        #   Store the results on Redis using the original job ID as the key
        #   so the API can match the results it gets to the original job
        #   sent
        
        # REDIS --> string thats why we need json.dumps:    
        db.set(data_dictionary['id'], json.dumps(prediction_dictionary)) 

        # Don't forget to sleep for a bit at the end
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
