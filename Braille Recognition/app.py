import cv2 
import tempfile
import os
import uuid
from flask import Flask, jsonify, render_template, send_file, redirect, request, Response
from werkzeug.utils import secure_filename
from OBR import SegmentationEngine, BrailleClassifier, BrailleImage

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
tempdir = tempfile.TemporaryDirectory()

app = Flask("Optical Braille Recognition Demo")
app.config['UPLOAD_FOLDER'] = tempdir.name

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/favicon.ico')
def fav():
    return send_file('favicon.ico', mimetype='image/ico')

@app.route('/coverimage')
def cover_image():
    return send_file('samples/sample1.png', mimetype='image/png')

@app.route('/procimage/<string:img_id>')
def proc_image(img_id):
    image = f"{tempdir.name}/{secure_filename(img_id)}-proc.png"
    if os.path.exists(image) and os.path.isfile(image):
        return send_file(image, mimetype='image/png')
    return redirect('/coverimage')

@app.route('/digest', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": True, "message": "file not in request"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": True, "message": "empty filename"})
    if file and allowed_file(file.filename):
        filename = ''.join(str(uuid.uuid4()).split('-'))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        classifier = BrailleClassifier()
        img = BrailleImage(image_path)
        for letter in SegmentationEngine(image=img):
            print(f"Segmented letter at position (top={letter.get_top()}, left={letter.get_left()})")
            letter.mark()
            classifier.push(letter)

        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}-proc.png")
        cv2.imwrite(processed_path, img.get_final_image())
        os.unlink(image_path)

        return jsonify({
            "error": False,
            "message": "Processed and Digested successfully",
            "img_id": filename,
            "digest": classifier.digest()
        })

@app.route('/webcam')
def webcam():
    return render_template("webcam.html")


@app.route('/video_feed')
def video_feed():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        filename = ''.join(str(uuid.uuid4()).split('-')) + ".jpg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(image_path, frame)

        classifier = BrailleClassifier()
        img = BrailleImage(image_path)

        for letter in SegmentationEngine(image=img):
            letter.mark()
            # 🔍 Add debug: show the raw dot info
            print(f"Segmented letter at position (top={letter.get_top()}, left={letter.get_left()})")

            classifier.push(letter)

        # 🔍 Show final decoded string
        print("Full Digest:", classifier.digest())

        proc_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}-proc.png")
        cv2.imwrite(proc_img_path, img.get_final_image())
        os.unlink(image_path)

        return jsonify({
            "error": False,
            "message": "Captured and processed",
            "img_id": filename,
            "digest": classifier.digest()
        })
    else:
        return jsonify({"error": True, "message": "Webcam capture failed"})



if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        tempdir.cleanup()


