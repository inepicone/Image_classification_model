from   curses             import flash
from   functools          import partial

import os
import hashlib
from   tkinter.messagebox import WARNING

def allowed_file(filename):
    """
    Checks if the format for the file received is acceptable. For this
    particular case, we must accept only image files. ['png', 'jpg', 'jpeg', 'gif']

    Parameters
    ----------
    filename : str
        Filename from werkzeug.datastructures.FileStorage file.

    Returns
    -------
    bool
        True if the file is an image, False otherwise.
    """
    # Current implementation will allow any kind of file.
    
    # formato de archivos permitidos:
    # usamos tuplas porque es mas rapido
    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}

    # _ variable de descarte
    ext = os.path.splitext(filename)[1].lower()

    # instaed of using if + else --> valid_file = ext in ALLOWED_EXTENSIONS esto me retorna true si esta en la lista que le pase o false si no esta
    valid_file = ext in ALLOWED_EXTENSIONS
    return valid_file

def get_file_hash(file):
    """
    Returns a new filename based on the file content using MD5 hashing.
    It uses hashlib.md5() function from Python standard library to get
    the hash.

    Parameters
    ----------
    file : werkzeug.datastructures.FileStorage
        File sent by user.

    Returns
    -------
    str
        New filename based in md5 file hash.
    """
    
    # Current implementation will return the original file name.
   
    md5 = hashlib.md5(file.read()).hexdigest()
    
    ext = os.path.splitext(file.filename)[1].lower()

    # return --> md5.hexdigest() + ext  
    ret = f'{md5}{ext}'
    # its important to go back to the forst position to start reading the next file:
    file.seek(0)
    
    return ret