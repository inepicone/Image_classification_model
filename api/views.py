from opcode import hasname
import os
import utils
import settings
import json
from werkzeug.utils import secure_filename
from middleware import model_predict
from flask import jsonify

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/", methods=["GET"])
def index():
    """
    Index endpoint, renders our HTML code.
    """
    return render_template("index.html")


@router.route("/", methods=["POST"])
def upload_image():
    """
    Function used in our frontend so we can upload and show an image.
    When it receives an image from the UI, it also calls our ML model to
    get and display the predictions.
    """
    # No file received, show basic UI
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    # File received but no filename is provided, show basic UI
    file = request.files["file"]
    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)

    # File received and it's an image, we must show it and get predictions
    if file and utils.allowed_file(file.filename):
        # Get an unique file name using utils.get_file_hash() function
        # Store the image to disk using the new name
        # Send the file to be processed by the `model` service
        # Update `context` dict with the corresponding values

        hash_name = utils.get_file_hash(file)
        file_path = os.path.join(settings.UPLOAD_FOLDER, hash_name)
        if os.path.isfile(file_path) == False:
            hash_name = secure_filename(hash_name)
            file.save(file_path)
        predict, score = model_predict(hash_name)

        context = {
            "prediction": predict,
            "score": f"{score:.2%}",
            "filename": hash_name
        }

        # Update `render_template()` parameters as needed
        return render_template(
            "index.html", filename=hash_name, context=context
        )
    # File received and but it isn't an image
    else:
        flash("Allowed image types are -> png, jpg, jpeg, gif")
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
    # Check a file was sent and that file is an image
    # Store the image to disk
    # Send the file to be processed by the `model` service
    # Update and return `rpse` dict with the corresponding values
    # If user sends an invalid request (e.g. no file provided) this endpoint
    # should return `rpse` dict with default values HTTP 400 Bad Request code
    
    rpse = {"success": False, "prediction": None, "score": None}
    # file_check = ("file" in request.files) & (request.files["file"] is not None)

    if not "file" in request.files:
        return jsonify(rpse), 400
    
    file = request.files["file"]
    if request.files["file"] is None:
        return jsonify(rpse), 400

    if utils.allowed_file(file.filename):
        hash_name = utils.get_file_hash(file)
        file_path = os.path.join(settings.UPLOAD_FOLDER, hash_name)
        if os.path.isfile(file_path) == False:
            hash_name = secure_filename(hash_name)
            file.save(file_path)
        predict, score = model_predict(hash_name)
        rpse = {"success": True, "prediction": predict, "score": score}
        return jsonify(rpse), 200

    else:
        return jsonify(rpse), 400


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
    report = json.dumps(report)
    report = json.loads(report)

    # Store the reported data to a file on the corresponding path
    # already provided in settings.py module
    with open(settings.FEEDBACK_FILEPATH, "a") as data:
        data.write(str(report) + "\n")
        data.close()

    return render_template("index.html")
