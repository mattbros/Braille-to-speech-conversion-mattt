from app import get_combination, get_distance

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
        'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '0'
    }.get(value, '?')

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

        end, start, width, combo = get_combination(box, dots, diameter)
        print("🔢 Dot combination:", combo)

        if combo not in self.symbol_table:
            self.result += "*"
            return

        if self.prev_end is not None:
            dist = get_distance(self.prev_end, start)
            if dist * 0.5 > (width ** 2):
                self.result += " "
        self.prev_end = end

        symbol = self.symbol_table[combo]
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
