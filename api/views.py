import  json
import  utils
import  settings
import  os
from    middleware import model_predict

from flask import (
    Blueprint,          # blueprint es una forma de dividir el codigo.
    flash,              # es el print de flask.
    redirect,           # para llevar el usuario de una pagina a otra.
    render_template,    # para poder escribir html.
    request,            # se para en el puerto y se obtiene la informacion que puede obtener de cada ruta (cada endpoint)
    url_for,            # Esta operación recibe como parámetro el nombre del método y nos devuelve la ruta.
    jsonify
)

router = Blueprint("app_router", __name__, template_folder="templates")

@router.route("/", methods=["GET"])
def index():
    """
    Index endpoint, renders our HTML code.
    """
    return render_template("index.html")

#endopoint to upload de image we want to identify 
@router.route("/", methods=["POST"]) 
def upload_image():
    """
    Function used in our frontend so we can upload and show an image.
    When it receives an image from the UI, 
    it also calls our ML model to
    get and display the predictions.
    
    """
    
    # UI --> user interface
    # No file received, show basic UI

    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url) # if there is no file --> se queda en el mismo lugar?
    
    file = request.files["file"]
    
    # File received but no filename is provided, show basic UI
    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)
 
    # File received and it's an image, we must show it and get predictions
    if file and utils.allowed_file(file.filename):
        new_name    = utils.get_file_hash(file)
        path        = os.path.join(settings.UPLOAD_FOLDER, new_name)
        
        if not os.path.exists(path):
            file.save(path)

        # 3. Send the file to be processed by the `model` service
        predict, score = model_predict(new_name)

        #4. Update `context` dict with the corresponding values                     
        context = {
            "prediction":   predict,
            "score":        score,
            "filename":     new_name,
        }

        # Update `render_template()` parameters as needed
        
        return render_template(
            "index.html", filename = new_name, context = context
        )
    # File received and but it isn't an image
    else:
        #flash("Allowed image types are -> png, jpg, jpeg, gif")
        return redirect(request.url)


@router.route("/display/<filename>")
def display_image(filename):
    """
    Display uploaded image in our UI.
    """
    return redirect(
        url_for("static", filename="uploads/" + filename), code=301
    )


@router.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint used to get predictions without need to access the UI.

    Parameters
    ----------
    file : str
        Input image we want to get predictions from.

    Returns
    -------
    flask.Response
        JSON response from our API having the following format:
            {
                "success": bool,
                "prediction": str,
                "score": float,
            }

        - "success" will be True if the input file is valid and we get a
          prediction from our ML model.
        - "prediction" model predicted class as string.
        - "score" model confidence score for the predicted class as float.
    """
    # first lets create de json file for the error cases:
    dic_error_rpse = {"success": False, "prediction": None, "score": None}

    # if there is no file recives:
    if "file" not in request.files:
        return jsonify(dic_error_rpse),400

    # File received but without a name:
    file = request.files["file"]
    if file.filename == "":
        return jsonify(dic_error_rpse),400

    # Good cases: 
    #   1. Check a file was sent and that file is an image:
       
    if file and utils.allowed_file(file.filename) :
        new_name    = utils.get_file_hash(file)
        path        = os.path.join(settings.UPLOAD_FOLDER, new_name)
        
        if not os.path.exists(path):
            file.save(path)

    #   3. Send the file to be processed by the `model` service
        predict, score = model_predict(new_name)

    #   4. Update and return `rpse` dict with the corresponding values
        rpse = {
                "success":      True, 
                "prediction":   predict, 
                "score":        score
                }
        return jsonify(rpse),200
    # If user sends an invalid request (e.g. no file provided) this endpoint
    # should return `rpse` dict with default values HTTP 400 Bad Request code
   
    else:
        return jsonify(dic_error_rpse),400
    
@router.route("/feedback", methods=["GET", "POST"])
def feedback():
    """
    Store feedback from users about wrong predictions on a text file.

    Parameters
    ----------
    report : request.form
        Feedback given by the user with the following JSON format:
            {
                "filename": str,
                "prediction": str,
                "score": float
            }

        - "filename" corresponds to the image used stored in the uploads
          folder.
        - "prediction" is the model predicted class as string reported as
          incorrect.
        - "score" model confidence score for the predicted class as float.
    """
    # Get reported predictions from `report` key
    report = request.form.get("report")
    # if there is a report, then open it and whrite 
    if report:
        with open(
            settings.FEEDBACK_FILEPATH,'a+'
        ) as f:
            f.write(report+'\n')
            # \n is necessary to read it nicely
    
    return render_template("index.html")
