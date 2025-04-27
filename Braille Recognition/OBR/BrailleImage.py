import cv2
import numpy as np

class BrailleImage(object):
    def __init__(self, image_path):
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise IOError('Cannot open given image')

        self.correct_perspective()
        self.original = self.preprocess_image(self.original)

        gray = self.to_gray(self.original)

        self.edged_binary_image = self.__get_edged_binary_image(gray)
        self.binary_image = self.__get_binary_image(gray)

        self.final = self.original.copy()
        shape = self.original.shape
        self.height, self.width = shape[0], shape[1]
        self.channels = shape[2] if len(shape) == 3 else 1

    def to_gray(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image.copy()

    def preprocess_image(self, image):
        gray = self.to_gray(image)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        median = cv2.medianBlur(blur, 5)

        thresh = cv2.adaptiveThreshold(
            median,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            2,
        )

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        return closing

    def correct_perspective(self):
        gray = self.to_gray(self.original)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        if len(approx) != 4:
            return

        pts = approx.reshape(4, 2)
        rect = self.__order_points(pts)

        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        if maxWidth <= 0 or maxHeight <= 0:
            return

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.original, M, (maxWidth, maxHeight))

        self.original = warped
        shape = warped.shape
        self.height, self.width = shape[0], shape[1]
        self.channels = shape[2] if len(shape) == 3 else 1

    def __order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

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
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thres = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            19,
            -3
        )
        blur2 = cv2.medianBlur(thres, 5)
        _, th2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur3 = cv2.GaussianBlur(th2, (5, 5), 0)
        _, th3 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(th3)

    def __get_binary_image(self, gray):
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur2 = cv2.medianBlur(th2, 5)
        _, th3 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(th3)

