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

# --- Added Matching Utilities ---
def get_distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def get_dot_nearest(dots, diameter, pt1):
    nearest = None
    min_dist = float('inf')
    tolerance = (diameter * 1.25) ** 2
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
    end = (right, midpointY)
    start = (left, midpointY)
    width = right - left

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

        print(f"👉 Checking corner: {corner}, assigned pos {pos}")
        D = get_dot_nearest(local_dots, diameter, corner)
        if D is not None:
            print(f"✅ Found dot near {corner}: {D}")
            local_dots.remove(D)
            result[pos - 1] = 1
        else:
            print(f"❌ No dot near {corner}")
        if not local_dots:
            print("🚫 No more dots left to match.")
            break

    print("🧪 Final result array (dot combo):", result, "| Types:", [type(v) for v in result])
    return end, start, width, tuple(result)

# --- Flask setup ---
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

        global global_img_debug
        classifier = BrailleClassifier()
        img = BrailleImage(image_path)
        global_img_debug = img.get_original_image().copy()

        for letter in custom_segmentation(img):
            print("Character bounding box:", letter.get_bounding_box())
            print("Dots in this box:", letter.get_dot_coordinates())
            letter.mark()
            classifier.push(letter)

        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}-proc.png")
        cv2.imwrite(processed_path, img.get_final_image())
        os.unlink(image_path)

        print("Full Digest:", classifier.digest())

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

        global global_img_debug
        classifier = BrailleClassifier()
        img = BrailleImage(image_path)
        global_img_debug = img.get_original_image().copy()

        for letter in custom_segmentation(img):
            print("Character bounding box:", letter.get_bounding_box())
            print("Dots in this box:", letter.get_dot_coordinates())
            letter.mark()
            classifier.push(letter)

        proc_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}-proc.png")
        cv2.imwrite(proc_img_path, img.get_final_image())
        os.unlink(image_path)

        print("Full Digest:", classifier.digest())

        return jsonify({
            "error": False,
            "message": "Captured and processed",
            "img_id": filename,
            "digest": classifier.digest()
        })
    else:
        return jsonify({"error": True, "message": "Webcam capture failed"})

# --- Segmentation Logic ---
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

def custom_segmentation(image):
    print("⚙️ Running custom segmentation...")

    img = image.get_original_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if 3 <= radius <= 20:
            dots.append(((int(x), int(y)), int(radius)))

    print(f"🟣 Total detected dots: {len(dots)}")

    dots = sorted(dots, key=lambda d: (d[0][1], d[0][0]))
    line_threshold = 40
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
    dot_diameter = np.mean([d[1] * 2 for d in dots]) if dots else 10

    for line_num, line in enumerate(lines):
        line = sorted(line, key=lambda d: d[0][0])
        i = 0
        while i + 1 < len(line):
            group = line[i:i + 2]
            x_coords = [p[0][0] for p in group]
            y_top = min(p[0][1] for p in group) - int(dot_diameter)
            y_bot = max(p[0][1] for p in group) + int(dot_diameter)
            x_left = min(x_coords) - int(dot_diameter)
            x_right = max(x_coords) + int(dot_diameter)

            box = (x_left, x_right, y_top, y_bot)

            cell_dots = []
            for d in dots:
                dx, dy = d[0]
                if x_left <= dx <= x_right and y_top <= dy <= y_bot:
                    cell_dots.append(d)

            if len(cell_dots) >= 1:
                print(f"📦 Grouping {len(cell_dots)} dots into one character at {box}")
                characters.append(FakeBrailleCharacter(box, cell_dots, dot_diameter * 1.5))

            i += 2

    print(f"✅ Total Braille cells formed: {len(characters)}\n")
    return characters

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        tempdir.cleanup()
