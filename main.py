from flask import Flask, render_template, Response
from camera import VideoCamera
import cv2

app = Flask(__name__)

@app.route("/") 
def my_index():
    return render_template("index.html", flask_token="Hello   world")

def gen(camera):        
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame
            + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),   
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/close-webcam', methods=['POST'])
def close_webcam():
    global camera
    if camera is not None:
        print('Hello world!')
        camera.release()
        cv2.VideoCapture(0).video.releast()
        camera = None
        cv2.destroyAllWindows()
        return 'Webcam closed'
    else:
        return 'Webcam not closed'
    
app.run(debug=True)