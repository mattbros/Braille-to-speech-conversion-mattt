# 🔧 File: OBR/BrailleClassifier.py

from math import sqrt

def get_distance(p1, p2):
        x1,y1 = p1
        x2,y2 = p2
        return ((x2 - x1)**2) + ((y2 - y1)**2)

def get_dot_nearest(dots, diameter, pt1):
        nearest = None
        diameter **= 2
        for dot in dots:
            point = dot[0]
            dist_from_pt1 = get_distance(point, pt1)
            if dist_from_pt1 <= diameter:
                if nearest is None:
                    nearest = dot
                else:
                    pt = nearest[0]
                    ndist_from_pt1 = get_distance(pt, pt1)
                    if ndist_from_pt1 >= dist_from_pt1:
                        nearest = dot
        return nearest

def get_combination(box, dots, diameter, img_debug=None):
        import cv2

        result = [0, 0, 0, 0, 0, 0]
        left, right, top, bottom = box

        midpointY = int((bottom - top) / 2)
        end = (right, midpointY)
        start = (left, midpointY)
        width = int(right - left)

        corners = {
                (left, top): 1,
                (left, top + midpointY): 2,
                (left, bottom): 3,
                (right, top): 4,
                (right, top + midpointY): 5,
                (right, bottom): 6
        }

        for corner, pos in corners.items():
                if img_debug is not None:
                        cv2.circle(img_debug, corner, 6, (0, 0, 255), -1)

                print(f"👉 Checking corner: {corner}, assigned pos {pos}")
                D = get_dot_nearest(dots, int(diameter), corner)
                if D is not None:
                        print(f"✅ Found dot near {corner}: {D}")
                        dots.remove(D)
                        result[pos - 1] = 1
                else:
                        print(f"❌ No dot near {corner}")
                if len(dots) == 0:
                        print("🚫 No more dots left to match.")
                        break

        print("🧪 Final result array (dot combo):", result, "| Types:", [type(v) for v in result])
        return end, start, width, tuple(result)

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

def translate_to_number(value):
    return {
        'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5',
        'f': '6', 'g': '7', 'h': '8', 'i': '9'
    }.get(value, '0')

class BrailleClassifier:
    symbol_table = {
        (1,0,0,0,0,0): Symbol('a', letter=True),
        (1,1,0,0,0,0): Symbol('b', letter=True),
        (1,0,0,1,0,0): Symbol('c', letter=True),
        (1,0,0,1,1,0): Symbol('d', letter=True),
        (1,0,0,0,1,0): Symbol('e', letter=True),
        (1,1,0,1,0,0): Symbol('f', letter=True),
        (1,1,0,1,1,0): Symbol('g', letter=True),
        (1,1,0,0,1,0): Symbol('h', letter=True),
        (0,1,0,1,0,0): Symbol('i', letter=True),
        (0,1,0,1,1,0): Symbol('j', letter=True),
        (1,0,1,0,0,0): Symbol('k', letter=True),
        (1,1,1,0,0,0): Symbol('l', letter=True),
        (1,0,1,1,0,0): Symbol('m', letter=True),
        (1,0,1,1,1,0): Symbol('n', letter=True),
        (1,0,1,0,1,0): Symbol('o', letter=True),
        (1,1,1,1,0,0): Symbol('p', letter=True),
        (1,1,1,1,1,0): Symbol('q', letter=True),
        (1,1,1,0,1,0): Symbol('r', letter=True),
        (0,1,1,1,0,0): Symbol('s', letter=True),
        (0,1,1,1,1,0): Symbol('t', letter=True),
        (1,0,1,0,0,1): Symbol('u', letter=True),
        (1,1,1,0,0,1): Symbol('v', letter=True),
        (0,1,0,1,1,1): Symbol('w', letter=True),
        (1,0,1,1,0,1): Symbol('x', letter=True),
        (1,0,1,1,1,1): Symbol('y', letter=True),
        (1,0,1,0,1,1): Symbol('z', letter=True),
        (0,0,1,1,1,1): Symbol('#', special=True),
    }

    def __init__(self, img_debug=None):
        self.result = ''
        self.shift_on = False
        self.prev_end = None
        self.number = False
        self.img_debug = img_debug

    def push(self, character):
        if not character.is_valid():
            return
        box = character.get_bounding_box()
        dots = character.get_dot_coordinates()
        diameter = character.get_dot_diameter()
        end, start, width, combination = get_combination(box, dots, diameter, img_debug=self.img_debug)

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
            self.result += symbol.value
        elif symbol.value == '#':
            self.number = True

    def digest(self):
        return self.result

    def clear(self):
        self.result = ''
        self.shift_on = False
        self.prev_end = None
        self.number = False
