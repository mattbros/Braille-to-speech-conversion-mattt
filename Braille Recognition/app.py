import io
import os
from flask import Flask, request, jsonify, send_file, render_template, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import uuid
import tempfile

# Import the other modules
from BrailleImage import BrailleImage
from BrailleClassifier import BrailleClassifier
from SegmentationEngine import SegmentationEngine
from BrailleCharacter import BrailleCharacter  # Import the BrailleCharacter class

app = Flask(__name__)
tempdir = tempfile.TemporaryDirectory()
app.config['UPLOAD_FOLDER'] = tempdir.name
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Shared image for debug drawing in get_combination
global_img_debug = None


# --- Utility Functions ---
def get_distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def get_dot_nearest(dots, diameter, pt1):
    nearest = None
    min_dist = float('inf')
    tolerance = (diameter * 1.5) ** 2  # tolerance was 1.25, changed for more robustness
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

    local_dots = list(dots)  # Don't mutate original
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


def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(file):
    """
    Process the uploaded image to extract Braille characters.

    Args:
        file: The file object from the Flask request.

    Returns:
        list: A list of BrailleCharacter objects, or an error message.
    """
    try:
        # Read the image data from the file object
        filestr = file.read()
        # convert to numpy array
        file_bytes = np.frombuffer(filestr, np.uint8)
        # decode image
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Create a BrailleImage object
        braille_image = BrailleImage(image=img)

        # Initialize the segmentation engine
        segmentation_engine = SegmentationEngine(braille_image)

        # Get the segmented Braille characters
        segmented_characters = segmentation_engine.get_segmented_characters()

        # Initialize the Braille classifier
        braille_classifier = BrailleClassifier()

        # Classify the characters and get the Braille text
        braille_text = ""
        recognized_characters = []  # To store BrailleCharacter objects with recognized letters
        for character in segmented_characters:
            letter = braille_classifier.classify(character)
            braille_text += letter
            character.set_letter(letter)  # set the letter for each BrailleCharacter object
            recognized_characters.append(character)  # Add to the list
        return recognized_characters

    except Exception as e:
        return str(e)  # Return the error message as a string


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/webcam')
def webcam():
    return render_template("webcam.html")


@app.route('/capture', methods=['POST'])
def capture():
    """
    Endpoint to handle image capture from the webcam.
    Returns:
        json: A JSON response containing the processing status and recognized data.
    """
    if 'image' not in request.files:
        return jsonify({"error": True, "message": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": True, "message": "Empty filename"}), 400

    if file and allowed_file(file.filename):
        filename = ''.join(str(uuid.uuid4()).split('-')) + ".png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        global global_img_debug

        try:
            img = BrailleImage(image_path)
            global_img_debug = img.get_original_image().copy()
            segmentation_engine = SegmentationEngine(img)
            classifier = BrailleClassifier()

            recognized_characters = []
            braille_text = ""
            for char in segmentation_engine:
                char.mark()
                bbox = char.get_bounding_box()
                dot_coords = char.get_dot_coordinates()
                dot_diameter = char.get_dot_diameter()

                print(f"Bounding Box: {bbox}")
                print(f"Dot Coordinates: {dot_coords}")
                print(f"Dot Diameter: {dot_diameter}")

                end, start, width, combo = get_combination(bbox, dot_coords, dot_diameter)
                print(f"Combination: {combo}")
                letter = classifier.classify(char)
                braille_text += letter
                char.set_letter(letter)
                recognized_characters.append(char)

            os.unlink(image_path)  # Clean up the image file

            response_data = {
                "error": False,
                "message": "Success",
                "digest": braille_text,  # Changed to braille_text
                "recognized_characters": [
                    {
                        'bounding_box': c.get_bounding_box(),
                        'letter': c.get_letter(),
                    } for c in recognized_characters
                ]
            }
            return jsonify(response_data), 200  # Return JSON data

        except Exception as e:
            os.unlink(image_path)
            return jsonify({"error": True, "message": f"Error processing image: {str(e)}"}), 500
    else:
        return jsonify({"error": True, "message": "Invalid file format"}), 400



@app.route('/procimage/<string:img_id>')
def proc_image(img_id):
    image = os.path.join(app.config['UPLOAD_FOLDER'], f"{secure_filename(img_id)}-proc.png")
    if os.path.exists(image):
        return send_file(image, mimetype='image/png')
    return redirect('/coverimage')


@app.route('/digest', methods=['POST'])
def upload():
    """
    Endpoint to handle image uploads for Braille character recognition.
    Returns:
        json: A JSON response containing the processing status and recognized data.
    """
    if 'file' not in request.files:
        return jsonify({"error": True, "message": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": True, "message": "No selected file"})

    if file and allowed_file(file.filename):
        filename = ''.join(str(uuid.uuid4()).split('-')) + ".png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        global global_img_debug

        try:
            img = BrailleImage(image_path)
            global_img_debug = img.get_original_image().copy()
            segmentation_engine = SegmentationEngine(img)
            classifier = BrailleClassifier()

            recognized_characters = []
            braille_text = ""
            for char in segmentation_engine:
                char.mark()
                bbox = char.get_bounding_box()
                dot_coords = char.get_dot_coordinates()
                dot_diameter = char.get_dot_diameter()

                print(f"Bounding Box: {bbox}")
                print(f"Dot Coordinates: {dot_coords}")
                print(f"Dot Diameter: {dot_diameter}")

                end, start, width, combo = get_combination(bbox, dot_coords, dot_diameter)
                print(f"Combination: {combo}")
                letter = classifier.classify(char)
                braille_text += letter
                char.set_letter(letter)
                recognized_characters.append(char)

            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}-proc.png")
            cv2.imwrite(processed_path, img.get_final_image())
            os.unlink(image_path)  # Clean up the original image file

            response_data = {
                "error": False,
                "message": "Success",
                "img_id": filename,
                "digest": braille_text,  # Changed to braille_text
                "recognized_characters": recognized_characters
            }
            return jsonify(response_data), 200

        except Exception as e:
            os.unlink(image_path)  # Clean up the image file
            return jsonify({"error": True, "message": f"Error processing image: {str(e)}"}), 500
    else:
        return jsonify({"error": True, "message": "Invalid file format"}), 400


@app.route('/video_feed')
def video_feed():
    """
    Endpoint to provide a video feed for real-time Braille recognition.

    Yields:
        bytes: A JPEG encoded frame of the video.
    """
    def gen_frames():
        cap = cv2.VideoCapture(0)  # Use 0 for the default camera
        while True:
            success, frame = cap.read()
            if not success:
                break  # Exit the loop if we cannot read a frame

            # You can add image processing here if needed, but it's better to do the heavy lifting
            # in a separate thread or process to avoid blocking the video feed.  For example,
            # you could send the frame to a queue and process it in the background.

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()  # Release the camera

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        tempdir.cleanup()

