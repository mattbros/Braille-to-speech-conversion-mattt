from math import sqrt
import cv2

# Import global for optional debug drawing
try:
    from app import global_img_debug
except ImportError:
    global_img_debug = None


def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2) + ((y2 - y1) ** 2)


def get_dot_nearest(dots, diameter, pt1):
    nearest = None
    diameter_squared = diameter ** 2
    for dot in dots:
        point = dot[0]
        dist_from_pt1 = get_distance(point, pt1)
        if dist_from_pt1 <= diameter_squared:
            if nearest is None or get_distance(nearest[0], pt1) > dist_from_pt1:
                nearest = dot
    return nearest


def get_combination(box, dots, diameter):
    from app import global_img_debug

    result = [0] * 6
    left, right, top, bottom = box
    width = right - left
    height = bottom - top

    # Calculate expected positions of 6-dot Braille grid using relative spacing
    positions = [
        (left + 0.25 * width, top + 0.2 * height),   # Dot 1
        (left + 0.25 * width, top + 0.5 * height),   # Dot 2
        (left + 0.25 * width, top + 0.8 * height),   # Dot 3
        (left + 0.75 * width, top + 0.2 * height),   # Dot 4
        (left + 0.75 * width, top + 0.5 * height),   # Dot 5
        (left + 0.75 * width, top + 0.8 * height),   # Dot 6
    ]

    radius_squared = (diameter * 1.2) ** 2  # allow some wiggle room

    for idx, expected_center in enumerate(positions):
        for dot in dots:
            dot_center = dot[0]
            dist = get_distance(dot_center, expected_center)
            if dist < radius_squared:
                result[idx] = 1
                if global_img_debug is not None:
                    cv2.circle(global_img_debug, expected_center, 6, (0, 255, 0), -1)  # green = hit
                break
        else:
            if global_img_debug is not None:
                cv2.circle(global_img_debug, expected_center, 6, (0, 0, 255), -1)  # red = miss

    end = (right, (top + bottom) // 2)
    start = (left, (top + bottom) // 2)
    return end, start, width, tuple(result)



def translate_to_number(value):
    return {
        'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5',
        'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '0'
    }.get(value, '?')


class Symbol:
    def __init__(self, value=None, letter=False, special=False):
        self.is_letter = letter
        self.is_special = special
        self.value = value

    def is_valid(self):
        return self.value is not None and (self.is_letter or self.is_special)

    def letter(self):
        return self.is_letter

    def special(self):
        return self.is_special


class BrailleClassifier:
    symbol_table = {
        (1, 0, 0, 0, 0, 0): Symbol('a', letter=True),
        (1, 1, 0, 0, 0, 0): Symbol('b', letter=True),
        (1, 0, 0, 1, 0, 0): Symbol('c', letter=True),
        (1, 0, 0, 1, 1, 0): Symbol('d', letter=True),
        (1, 0, 0, 0, 1, 0): Symbol('e', letter=True),
        (1, 1, 0, 1, 0, 0): Symbol('f', letter=True),
        (1, 1, 0, 1, 1, 0): Symbol('g', letter=True),
        (1, 1, 0, 0, 1, 0): Symbol('h', letter=True),
        (0, 1, 0, 1, 0, 0): Symbol('i', letter=True),
        (0, 1, 0, 1, 1, 0): Symbol('j', letter=True),
        (1, 0, 1, 0, 0, 0): Symbol('k', letter=True),
        (1, 1, 1, 0, 0, 0): Symbol('l', letter=True),
        (1, 0, 1, 1, 0, 0): Symbol('m', letter=True),
        (1, 0, 1, 1, 1, 0): Symbol('n', letter=True),
        (1, 0, 1, 0, 1, 0): Symbol('o', letter=True),
        (1, 1, 1, 1, 0, 0): Symbol('p', letter=True),
        (1, 1, 1, 1, 1, 0): Symbol('q', letter=True),
        (1, 1, 1, 0, 1, 0): Symbol('r', letter=True),
        (0, 1, 1, 1, 0, 0): Symbol('s', letter=True),
        (0, 1, 1, 1, 1, 0): Symbol('t', letter=True),
        (1, 0, 1, 0, 0, 1): Symbol('u', letter=True),
        (1, 1, 1, 0, 0, 1): Symbol('v', letter=True),
        (0, 1, 0, 1, 1, 1): Symbol('w', letter=True),
        (1, 0, 1, 1, 0, 1): Symbol('x', letter=True),
        (1, 0, 1, 1, 1, 1): Symbol('y', letter=True),
        (1, 0, 1, 0, 1, 1): Symbol('z', letter=True),
        (0, 0, 1, 1, 1, 1): Symbol('#', special=True),
    }

    def __init__(self):
        self.result = ''
        self.shift_on = False
        self.prev_end = None
        self.number = False

    def push(self, character):
        if not character.is_valid():
            return

        box = character.get_bounding_box()
        dots = character.get_dot_coordinates()
        diameter = character.get_dot_diameter()

        end, start, width, combination = get_combination(box, dots, diameter)
        print("🔢 Dot combination:", combination)

        if combination not in self.symbol_table:
            self.result += "*"
            return

        if self.prev_end is not None:
            dist = get_distance(self.prev_end, start)
            if dist * 0.5 > (width ** 2):
                self.result += " "
        self.prev_end = end

        symbol = self.symbol_table[combination]
        if symbol.letter() and self.number:
            self.number = False
            self.result += translate_to_number(symbol.value)
        elif symbol.letter():
            self.result += symbol.value.upper() if self.shift_on else symbol.value
        elif symbol.special() and symbol.value == '#':
            self.number = True

    def digest(self):
        return self.result

    def clear(self):
        self.result = ''
        self.shift_on = False
        self.prev_end = None
        self.number = False
