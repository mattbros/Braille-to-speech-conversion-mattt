import cv2
import tempfile
import os
import uuid
import numpy as np
from flask import Flask, jsonify, render_template, send_file, redirect, request, Response
from werkzeug.utils import secure_filename
from OBR import SegmentationEngine, BrailleClassifier, BrailleImage

# Shared image for debug drawing in get_combination
global_img_debug = None

app = Flask("Optical Braille Recognition Demo")
tempdir = tempfile.TemporaryDirectory()
app.config['UPLOAD_FOLDER'] = tempdir.name
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Utility Functions ---
def get_distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def get_dot_nearest(dots, diameter, pt1):
    nearest = None
    min_dist = float('inf')
    tolerance = (diameter * 1.5) ** 2
    for dot in dots:
        dist = get_distance(dot[0], pt1)
        if dist <= tolerance and dist < min_dist:
            nearest = dot
            min_dist = dist
    return nearest

def get_combination(box, dots, diameter):
    global global_img_debug

    result = [0, 0, 0, 0, 0, 0]
    left, right, top, bottom = box
    midpointY = (bottom - top) // 2
    corners = {
        (left, top): 1,
        (left, top + midpointY): 2,
        (left, bottom): 3,
        (right, top): 4,
        (right, top + midpointY): 5,
        (right, bottom): 6
    }

    local_dots = list(dots)
    for corner, pos in corners.items():
        if global_img_debug is not None:
            cv2.circle(global_img_debug, corner, 6, (0, 0, 255), -1)

        D = get_dot_nearest(local_dots, diameter, corner)
        if D is not None:
            local_dots.remove(D)
            result[pos - 1] = 1
        if not local_dots:
            break

    return (right, midpointY), (left, midpointY), right - left, tuple(result)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/webcam')
def webcam():
    return render_template("webcam.html")

@app.route('/capture', methods=['POST'])
def capture():
    if 'image' not in request.files:
        return jsonify({"error": True, "message": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": True, "message": "Empty filename"}), 400

    if file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        filename = ''.join(str(uuid.uuid4()).split('-')) + ".png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        from OBR import BrailleImage, BrailleClassifier, SegmentationEngine
        global global_img_debug

        try:
            img = BrailleImage(image_path)
        except Exception as e:
            return jsonify({"error": True, "message": f"Image processing error: {str(e)}"}), 500

        global_img_debug = img.get_original_image().copy()
        segmentation_engine = SegmentationEngine(img)
        classifier = BrailleClassifier()

        for char in segmentation_engine:
            char.mark()
            bbox = char.get_bounding_box()
            dot_coords = char.get_dot_coordinates()
            dot_diameter = char.get_dot_diameter()

            end, start, width, combo = get_combination(bbox, dot_coords, dot_diameter)
            classifier.push(char)

        os.unlink(image_path)

        return jsonify({
            "error": False,
            "message": "Success",
            "digest": classifier.digest()
        })

    return jsonify({"error": True, "message": "Invalid image format"}), 400

@app.route('/procimage/<string:img_id>')
def proc_image(img_id):
    image = os.path.join(app.config['UPLOAD_FOLDER'], f"{secure_filename(img_id)}-proc.png")
    if os.path.exists(image):
        return send_file(image, mimetype='image/png')
    return redirect('/coverimage')

@app.route('/digest', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": True, "message": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": True, "message": "No selected file"})

    if file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        filename = ''.join(str(uuid.uuid4()).split('-'))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        from OBR import BrailleImage, BrailleClassifier, SegmentationEngine
        global global_img_debug

        try:
            img = BrailleImage(image_path)
        except Exception as e:
            return jsonify({"error": True, "message": f"Image processing error: {str(e)}"}), 500

        global_img_debug = img.get_original_image().copy()
        segmentation_engine = SegmentationEngine(img)
        classifier = BrailleClassifier()

        for char in segmentation_engine:
            char.mark()
            bbox = char.get_bounding_box()
            dot_coords = char.get_dot_coordinates()
            dot_diameter = char.get_dot_diameter()

            end, start, width, combo = get_combination(bbox, dot_coords, dot_diameter)
            classifier.push(char)

        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}-proc.png")
        cv2.imwrite(processed_path, img.get_final_image())
        os.unlink(image_path)

        return jsonify({
            "error": False,
            "message": "Success",
            "img_id": filename,
            "digest": classifier.digest()
        })

    return jsonify({"error": True, "message": "Invalid file format"})

@app.route('/video_feed')
def video_feed():
    def gen_frames():
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
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        tempdir.cleanup()
