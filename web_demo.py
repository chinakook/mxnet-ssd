from flask import Flask, request
#from werkzeug.utils import secure_filename
import hashlib
import cv2
import numpy as np
import datetime
from test_one import *

app = Flask(__name__)

det_mod = get_detection_mod()

cls_mod = get_classification_mod()

def mx_predict(file_location):
    fname = mx.test_utils.download(file_location, dirname="static/img_pool")
    raw_img = cv2.imread(fname)

    if raw_img is None:
        return ""

    dets = det_img(raw_img, det_mod)

    retstr = ""
    if dets is not None:
        xmin = int(dets[0])
        ymin = int(dets[1])
        xmax = int(dets[2])
        ymax = int(dets[3])
        roi_w = xmax - xmin + 1
        roi_h = ymax - ymin + 1

        if roi_w > roi_h:
            pad = roi_w - roi_h
            ymin = ymin - pad // 2
            ymax = ymax + (pad+1)//2
        else:
            pad = roi_h - roi_w
            xmin = xmin - pad // 2
            xmax = xmax + (pad+1)//2
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(raw_img.shape[1]-1, xmax)
            ymax = min(raw_img.shape[0]-1, ymax)

        roi_img = raw_img[ymin:ymax+1, xmin:xmax+1,:]

        ylb_type, pred = cls_img(roi_img, cls_mod)

        retstr = "%d, %f, (%d, %d, %d, %d)" % (ylb_type, pred[0, ylb_type], xmin, ymin, xmax, ymax)

    return retstr
    

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_image", methods = ['POST'])
def FUN_upload_image():

    if request.method == 'POST':
        print("img uploaded")
        # check if the post request has the file part
        if 'file' not in request.files:
            return ""
        file = request.files['file']

        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            return ""

        if file and allowed_file(file.filename):
            filename = os.path.join("static/img_pool", hashlib.sha256(str(datetime.datetime.now())).hexdigest() + file.filename.lower())
            filename=filename.replace('\\','/')
            file.save(filename)
            prediction_result = mx_predict(filename)
	        #FUN_resize_img(filename)
            return prediction_result

        return ""
            #return render_template("index.html", img_src = filename, prediction_result = prediction_result)
    #return(redirect(url_for("FUN_root")))

################################################
# Start the service
################################################
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)