import cv2
import numpy as np
from collections import Counter

class SegmentationEngine:
    def __init__(self, braille_image):
        self.img = braille_image.get_binary_image()
        self.height = braille_image.get_height()
        self.width = braille_image.get_width()
        self.enclosingCircles = self.__find_enclosing_circles()

        if not self.enclosingCircles:
            raise ValueError("No dots found in Braille image.")

        self.diameter, self.dots, self.radius = self.__get_valid_dots(self.enclosingCircles)
        self.chars = self.__segment_characters()
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.index >= len(self.chars):
            raise StopIteration
        char = self.chars[self.index]
        self.index += 1
        return char

    def __find_enclosing_circles(self):
        contours, _ = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enclosing_circles = []

        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 1:  # Only keep reasonable sized circles
                enclosing_circles.append((int(x), int(y), int(radius)))

        return enclosing_circles

    def __get_valid_dots(self, enclosingCircles):
        radii = [circle[2] for circle in enclosingCircles]

        if not radii:
            raise ValueError("No valid dot radii found.")

        # Find the most common radius
        counter = Counter(radii)
        if len(counter) == 0:
            raise ValueError("No dots detected.")

        baserad = counter.most_common(1)[0][0]

        # Filter dots that are close to the base radius
        tolerance = baserad * 0.5
        valid_dots = [(x, y) for (x, y, r) in enclosingCircles if abs(r - baserad) <= tolerance]

        return baserad * 2, valid_dots, baserad

