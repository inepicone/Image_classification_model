import  time
import  redis
import  settings
import  uuid         # this is to generate a Universal Unique Identifier
import  json         # esto es para usar los comandos que transforman json

# middleware is the one that connect the api with redis (and redis consct to the model)

# redis its a database in form of queue in this case:
db = redis.Redis(       
    host = settings.REDIS_IP, 
    port = settings.REDIS_PORT, 
    db   = settings.REDIS_DB_ID,
)

# Connect to Redis
#if condition returns False, AssertionError is raised:
# db.ping() --> to connect 
assert db.ping, "I couldnt connect"

def model_predict(image_name):
    """
    Receives an image name and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    image_name : str
        Name for the image uploaded by the user.

    Returns
    -------
    prediction, score : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    
    # Assign an unique ID for this job:
    job_id = str(uuid.uuid4())
    
    job_data = {
        "id": job_id,
        "image_name": image_name,
    }
           
    # Send the job to the model service using Redis
    # we transform it to json because redis doesnt operate with dic:
    db.rpush(settings.REDIS_QUEUE, json.dumps(job_data)) 
    
    prediction        = ''
    score             = 0

    # Loop until we received the response from our ML model
    while True:
        # Attempt to get model predictions using job_id
        # to get a value using a key from Redis --> get()
        # if the job_id exists then: get the results from the model.
        if (db.exists(job_id)): 
            
            output            = db.get(job_id)
            # now output its a string so lets change it to a dictionary:
            output            = json.loads(output) 
            # this dictionary have: clase and score
            
            prediction        = output['prediction'] 
            score             = output['score']
            
            # Because I have everything save at the variables I can delete it from redis:
            db.delete(job_id)
            # Then exit the loop
        
            break 
        
        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)


    return prediction, score
