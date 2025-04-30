# Optical Braille Recognition (OBR)

## Overview

This project is an Optical Braille Recognition (OBR) system that aims to recognize Braille characters from images. It can be used to assist visually impaired individuals in reading Braille text.

## Features

* **Image Input:** Accepts images as input.
* **Perspective Correction:** Corrects perspective distortions in the image.
* **Dot Segmentation:** Identifies and segments individual Braille dots.
* **Character Recognition:** Recognizes Braille characters from the dot patterns.
* **Output:** Provides the recognized Braille text as output.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Set up a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare an image:**

    * Take a clear picture of the Braille text.
    * Ensure the image is well-lit and the Braille text is in focus.
    * Save the image in a common format like JPG or PNG.

2.  **Run the application:**

    ```bash
    python app.py
    ```

3.  **Use the application:**

    * The application will process the image and output the recognized Braille text. The exact way to provide the image will depend on how you've set up the app (e.g., a command-line argument, a web interface). See the `app.py` file for specific usage details.

## Code Description

Key files and their roles:

* `app.py`: This is the main application file. It handles image input, calls the OBR processing functions, and outputs the recognized text. It likely uses Flask for handling web requests.
* `OBR/SegmentationEngine.py`: This file contains the `SegmentationEngine` class, which is responsible for segmenting the Braille dots from the image.
* `OBR/BrailleImage.py`: This file contains the `BrailleImage` class, which handles image loading, preprocessing (including perspective correction), and provides access to image data.
* `OBR/BrailleCharacter.py`: This file contains the `BrailleCharacter` class, which represents a single Braille character and its properties (dot coordinates, bounding box, etc.).
* `requirements.txt`: A list of Python packages required to run the application.

## Modifications

* `BrailleImage.py`:
    * Perspective correction is handled by the `correct_perspective` method.
    * Contour filtering is used to find the Braille region.
    * Key parameters that were tuned:
        * `approxPolyDP` parameter is set to 0.01 for more precise shape approximation.
        * Contour filtering thresholds are set to `0.05 < circularity < 0.95` and `area > 800` to be more inclusive and allow smaller contours.

## Dependencies

* OpenCV (`cv2`): For image processing.
* NumPy: For numerical operations.
* Flask: For creating a web application (if applicable).
* Other packages as listed in `requirements.txt`.

## License

This project is licensed under the **MIT License**.

## Acknowledgements

* I'd like to acknowledge the contributions of the open-source community, whose libraries and tools made this project possible.  Specifically:
    * **OpenCV:** The core library for image processing.
    * **NumPy:** For efficient numerical computation.
    * **Flask:** (If used) For providing a web framework.
