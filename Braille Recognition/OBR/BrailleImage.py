import cv2
import numpy as np

class BrailleImage(object):
    def __init__(self, image):
        # Read source image
        self.original = cv2.imread(image)
        if self.original is None:
            raise IOError('Cannot open given image')

        self.correct_perspective()  # <<< Perspective correction step added here

        # First Layer, Convert BGR(Blue Green Red Scale) to Gray Scale
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        # Save the binary image of the edge detected
        self.edged_binary_image = self.__get_edged_binary_image(gray)

        # Save a binary image to get the contents inside the edges
        self.binary_image = self.__get_binary_image(gray)

        self.final = self.original.copy()
        self.height, self.width, self.channels = self.original.shape
        return

    def correct_perspective(self):
        """Auto-correct the perspective tilt of the Braille image."""
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(blur, 50, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return  # No contours found; skip correction

        c = max(contours, key=cv2.contourArea)

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) != 4:
            return  # Not a quadrilateral; skip correction

        pts = approx.reshape(4, 2)
        rect = self.__order_points(pts)

        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.original, M, (maxWidth, maxHeight))

        self.original = warped
        self.height, self.width, self.channels = warped.shape

    def __order_points(self, pts):
        """Helper to consistently order corner points."""
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect

    def bound_box(self, left, right, top, bottom, color=(255, 0, 0), size=1):
        self.final = cv2.rectangle(self.final, (left, top), (right, bottom), color, size)
        return True

    def get_final_image(self):
        return self.final

    def get_original_image(self):
        return self.original

    def get_edged_binary_image(self):
        return self.edged_binary_image

    def get_binary_image(self):
        return self.binary_image

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def __get_edged_binary_image(self, gray):
        # First Lvl Blur to Reduce Noise - Even more aggressive and adaptive blurring
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive Thresholding to define the dots in Braille - More adaptive parameters
        thres = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            19,  # Increased block size even more
            -3   # Slightly more negative C
        )

        blur2 = cv2.medianBlur(thres, 5)
        ret2, th2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur3 = cv2.GaussianBlur(th2, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(th3)

    def __get_binary_image(self, gray):
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        ret2, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur2 = cv2.medianBlur(th2, 5)
        ret3, th3 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(th3)

