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
        self.grid_spacing_x, self.grid_spacing_y = self.__estimate_grid_spacing()
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
            if radius > 1:
                enclosing_circles.append((int(x), int(y), int(radius)))

        return enclosing_circles

    def __get_valid_dots(self, enclosingCircles):
        radii = [circle[2] for circle in enclosingCircles]

        if not radii:
            raise ValueError("No valid dot radii found.")

        counter = Counter(radii)
        if len(counter) == 0:
            raise ValueError("No dots detected.")

        baserad = counter.most_common(1)[0][0]
        tolerance = baserad * 0.5
        valid_dots = [(x, y) for (x, y, r) in enclosingCircles if abs(r - baserad) <= tolerance]

        return baserad * 2, valid_dots, baserad

    def __estimate_grid_spacing(self):
        if not self.dots:
            return 1, 1

        xs = [x for (x, y) in self.dots]
        ys = [y for (x, y) in self.dots]

        xs.sort()
        ys.sort()

        dx = np.median(np.diff(xs)) if len(xs) > 1 else 1
        dy = np.median(np.diff(ys)) if len(ys) > 1 else 1

        return max(dx, 1), max(dy, 1)

    def __segment_characters(self):
        if not self.dots:
            return []

        grid = {}
        for (x, y) in self.dots:
            grid_x = int(round(x / self.grid_spacing_x))
            grid_y = int(round(y / self.grid_spacing_y))
            grid[(grid_x, grid_y)] = (x, y)

        min_x = min(k[0] for k in grid.keys())
        max_x = max(k[0] for k in grid.keys())
        min_y = min(k[1] for k in grid.keys())
        max_y = max(k[1] for k in grid.keys())

        chars = []
        for gx in range(min_x, max_x + 1, 2):
            for gy in range(min_y, max_y + 1, 3):
                cell_dots = []
                for dx in range(2):
                    for dy in range(3):
                        if (gx + dx, gy + dy) in grid:
                            cell_dots.append(grid[(gx + dx, gy + dy)])
                if cell_dots:
                    chars.append(Character(cell_dots))

        return chars

class Character:
    def __init__(self, dots):
        self.dots = dots

    def mark(self):
        pass  # Placeholder for marking logic

    def get_bounding_box(self):
        xs = [dot[0] for dot in self.dots]
        ys = [dot[1] for dot in self.dots]
        return min(xs), max(xs), min(ys), max(ys)

    def get_dot_coordinates(self):
        return [(dot, 0) for dot in self.dots]

    def get_dot_diameter(self):
        return 5  # Placeholder value
