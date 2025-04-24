import cv2
import tempfile
import os
import uuid
import numpy as np
from flask import Flask, jsonify, render_template, send_file, redirect, request, Response
from werkzeug.utils import secure_filename
from OBR import SegmentationEngine, BrailleClassifier, BrailleImage

app = Flask("Optical Braille Recognition Demo")
tempdir = tempfile.TemporaryDirectory()
app.config['UPLOAD_FOLDER'] = tempdir.name
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Shared image for debug drawing in get_combination
global_img_debug = None

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
            cv2.circle(global_img_debug, corner, 6, (255, 0, 0), 2)
            cv2.putText(global_img_debug, str(pos), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        D = get_dot_nearest(local_dots, diameter, corner)
        print(f"🔵 Corner {corner} → Dot {D}")
        if D:
            result[pos - 1] = 1
            local_dots.remove(D)
            if global_img_debug is not None:
                cv2.circle(global_img_debug, D[0], 6, (0, 255, 0), -1)
        else:
            print(f"🟡 No dot found near {corner} (expected pos {pos})")

    print("🔢 Dot combination:", tuple(result))
    return None, None, None, tuple(result)

# --- Character Class ---
class FakeBrailleCharacter:
    def __init__(self, bounding_box, dot_coords, dot_diameter):
        self._bbox = bounding_box
        self._dots = dot_coords
        self._diameter = dot_diameter

    def get_bounding_box(self):
        return self._bbox

    def get_dot_coordinates(self):
        return self._dots

    def get_dot_diameter(self):
        return self._diameter

    def is_valid(self):
        return True

    def mark(self):
        pass

# --- Segmentation ---
def custom_segmentation(image):
    print("⚙️ Running custom segmentation...")

    img = image.get_original_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 3
    )

    cv2.imwrite("thresh_debug.png", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if 3 <= radius <= 18:
            dots.append(((int(x), int(y)), int(radius)))

    print(f"🔣 Total detected dots: {len(dots)}")

    dot_diameter = np.median([d[1] * 2 for d in dots]) if dots else 10
    dots = sorted(dots, key=lambda d: (d[0][1], d[0][0]))
    line_threshold = int(dot_diameter * 2.0)

    lines = []
    current_line = []

    for dot in dots:
        if not current_line or abs(dot[0][1] - current_line[-1][0][1]) < line_threshold:
            current_line.append(dot)
        else:
            lines.append(current_line)
            current_line = [dot]
    if current_line:
        lines.append(current_line)

    print(f"📏 Lines detected: {len(lines)}")

    characters = []

    for line_num, line in enumerate(lines):
        line = sorted(line, key=lambda d: d[0][0])
        i = 0
        while i < len(line):
            group = [line[i]]
            cx, cy = line[i][0]
            j = i + 1
            while j < len(line):
                nx, ny = line[j][0]
                if abs(nx - cx) < dot_diameter * 1.3:
                    group.append(line[j])
                    j += 1
                else:
                    break

            if len(group) > 6:
                print(f"⚠️ Skipping group of {len(group)} dots – likely overlapping characters")
                i += 1
                continue

            if 1 <= len(group) <= 6:
                x_coords = [p[0][0] for p in group]
                y_coords = [p[0][1] for p in group]
                x_left = min(x_coords) - int(dot_diameter)
                x_right = max(x_coords) + int(dot_diameter)
                y_top = min(y_coords) - int(dot_diameter)
                y_bot = max(y_coords) + int(dot_diameter)

                box = (x_left, x_right, y_top, y_bot)
                characters.append(FakeBrailleCharacter(box, group, dot_diameter * 1.5))

            i += len(group)

    print(f"✅ Total Braille cells formed: {len(characters)}\n")
    if global_img_debug is not None:
        cv2.imwrite("debug_overlay.png", global_img_debug)

    return characters


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/webcam')
def webcam():
    return render_template("webcam.html")

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

        classifier = BrailleClassifier()
        img = BrailleImage(image_path)
        global global_img_debug
        global_img_debug = img.get_original_image().copy()

        characters = custom_segmentation(img)
        for char in characters:
            char.mark()
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

        classifier = BrailleClassifier()
        img = BrailleImage(image_path)
        global global_img_debug
        global_img_debug = img.get_original_image().copy()

        characters = custom_segmentation(img)
        for char in characters:
            char.mark()
            classifier.push(char)

        os.unlink(image_path)

        return jsonify({
            "error": False,
            "message": "Success",
            "digest": classifier.digest()
        })

    return jsonify({"error": True, "message": "Invalid file format"})

if __name__ == "__main__":
    try:
        os.system("git pull origin mattbros-patch-1")
        app.run(debug=True)
    finally:
        tempdir.cleanup()
