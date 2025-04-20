from math import sqrt
import cv2

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
                if nearest is None:
                    nearest = dot
                else:
                    pt = nearest[0]
                    ndist_from_pt1 = get_distance(pt, pt1)
                    if ndist_from_pt1 >= dist_from_pt1:
                        nearest = dot
        return nearest

def get_combination(box, dots, diameter):
    import cv2
    global global_img_debug

    result = [0, 0, 0, 0, 0, 0]
    left, right, top, bottom = box

    cell_height = (bottom - top)
    cell_width = (right - left)
    third_height = cell_height // 3

    # Calculate positions for standard 6-dot Braille (top-down, left-to-right)
    corners = {
        (left, top): 1,                        # Top-left
        (left, top + third_height): 2,        # Middle-left
        (left, top + 2 * third_height): 3,    # Bottom-left
        (right, top): 4,                      # Top-right
        (right, top + third_height): 5,       # Middle-right
        (right, top + 2 * third_height): 6    # Bottom-right
    }

    for corner, pos in corners.items():
        cx, cy = corner
        if global_img_debug is not None:
            cv2.circle(global_img_debug, (cx, cy), 6, (0, 0, 255), -1)  # Red dot

        print(f"👉 Checking corner: {corner}, assigned pos {pos}")
        D = get_dot_nearest(dots, int(diameter * 1.5), corner)  # more generous match
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
    return (right, top + cell_height // 2), (left, top + cell_height // 2), cell_width, tuple(result)


def translate_to_number(value):
    return {
        'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5',
        'f': '6', 'g': '7', 'h': '8', 'i': '9'
    }.get(value, '0')

class Symbol(object):
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

class BrailleClassifier(object):
    symbol_table = {
         (1,0,0,0,0,0): Symbol('a',letter=True),
         (1,1,0,0,0,0): Symbol('b',letter=True),
         (1,0,0,1,0,0): Symbol('c',letter=True),
         (1,0,0,1,1,0): Symbol('d',letter=True),
         (1,0,0,0,1,0): Symbol('e',letter=True),
         (1,1,0,1,0,0): Symbol('f',letter=True),
         (1,1,0,1,1,0): Symbol('g',letter=True),
         (1,1,0,0,1,0): Symbol('h',letter=True),
         (0,1,0,1,0,0): Symbol('i',letter=True),
         (0,1,0,1,1,0): Symbol('j',letter=True),
         (1,0,1,0,0,0): Symbol('K',letter=True),
         (1,1,1,0,0,0): Symbol('l',letter=True),
         (1,0,1,1,0,0): Symbol('m',letter=True),
         (1,0,1,1,1,0): Symbol('n',letter=True),
         (1,0,1,0,1,0): Symbol('o',letter=True),
         (1,1,1,1,0,0): Symbol('p',letter=True),
         (1,1,1,1,1,0): Symbol('q',letter=True),
         (1,1,1,0,1,0): Symbol('r',letter=True),
         (0,1,1,1,0,0): Symbol('s',letter=True),
         (0,1,1,1,1,0): Symbol('t',letter=True),
         (1,0,1,0,0,1): Symbol('u',letter=True),
         (1,1,1,0,0,1): Symbol('v',letter=True),
         (0,1,0,1,1,1): Symbol('w',letter=True),
         (1,0,1,1,0,1): Symbol('x',letter=True),
         (1,0,1,1,1,1): Symbol('y',letter=True),
         (1,0,1,0,1,1): Symbol('z',letter=True),
         (0,0,1,1,1,1): Symbol('#',special=True),
    }

    def __init__(self):
        self.result = ''
        self.shift_on = False
        self.prev_end = None
        self.number = False

    def push(self, character):
        if not character.is_valid():
            return

        from app import global_img_debug  # safely import inside method

        box = character.get_bounding_box()
        dots = character.get_dot_coordinates()
        diameter = character.get_dot_diameter()

        end, start, width, combination = get_combination(box, dots, diameter, global_img_debug)
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
        else:
            if symbol.value == '#':
                self.number = True

    def digest(self):
        return self.result

    def clear(self):
        self.result = ''
        self.shift_on = False
        self.prev_end = None
        self.number = False

