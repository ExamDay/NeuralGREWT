import json

colorMap = {
    "maroon": (128, 0, 0),
    "dark red": (139, 0, 0),
    "brown": (165, 42, 42),
    "firebrick": (178, 34, 34),
    "crimson": (220, 20, 60),
    "red": (255, 0, 0),
    "tomato": (255, 99, 71),
    "coral": (255, 127, 80),
    "indian red": (205, 92, 92),
    "light coral": (240, 128, 128),
    "dark salmon": (233, 150, 122),
    "salmon": (250, 128, 114),
    "light salmon": (255, 160, 122),
    "orange red": (255, 69, 0),
    "dark orange": (255, 140, 0),
    "orange": (255, 165, 0),
    "gold": (255, 215, 0),
    "dark golden rod": (184, 134, 11),
    "golden rod": (218, 165, 32),
    "pale golden rod": (238, 232, 170),
    "dark khaki": (189, 183, 107),
    "khaki": (240, 230, 140),
    "olive": (128, 128, 0),
    "yellow": (255, 255, 0),
    "yellow green": (154, 205, 50),
    "dark olive green": (85, 107, 47),
    "olive drab": (107, 142, 35),
    "lawn green": (124, 252, 0),
    "chart reuse": (127, 255, 0),
    "green yellow": (173, 255, 47),
    "dark green": (0, 100, 0),
    "green": (0, 128, 0),
    "forest green": (34, 139, 34),
    "lime": (0, 255, 0),
    "lime green": (50, 205, 50),
    "light green": (144, 238, 144),
    "pale green": (152, 251, 152),
    "dark sea green": (143, 188, 143),
    "medium spring green": (0, 250, 154),
    "spring green": (0, 255, 127),
    "sea green": (46, 139, 87),
    "medium aqua marine": (102, 205, 170),
    "medium sea green": (60, 179, 113),
    "light sea green": (32, 178, 170),
    "dark slate gray": (47, 79, 79),
    "teal": (0, 128, 128),
    "dark cyan": (0, 139, 139),
    "aqua": (0, 255, 255),
    "cyan": (0, 255, 255),
    "light cyan": (224, 255, 255),
    "dark turquoise": (0, 206, 209),
    "turquoise": (64, 224, 208),
    "medium turquoise": (72, 209, 204),
    "pale turquoise": (175, 238, 238),
    "aqua marine": (127, 255, 212),
    "powder blue": (176, 224, 230),
    "cadet blue": (95, 158, 160),
    "steel blue": (70, 130, 180),
    "corn flower blue": (100, 149, 237),
    "deep sky blue": (0, 191, 255),
    "dodger blue": (30, 144, 255),
    "light blue": (173, 216, 230),
    "sky blue": (135, 206, 235),
    "light sky blue": (135, 206, 250),
    "midnight blue": (25, 25, 112),
    "navy": (0, 0, 128),
    "dark blue": (0, 0, 139),
    "medium blue": (0, 0, 205),
    "blue": (0, 0, 255),
    "royal blue": (65, 105, 225),
    "blue violet": (138, 43, 226),
    "indigo": (75, 0, 130),
    "dark slate blue": (72, 61, 139),
    "slate blue": (106, 90, 205),
    "medium slate blue": (123, 104, 238),
    "medium purple": (147, 112, 219),
    "dark magenta": (139, 0, 139),
    "dark violet": (148, 0, 211),
    "dark orchid": (153, 50, 204),
    "medium orchid": (186, 85, 211),
    "purple": (128, 0, 128),
    "thistle": (216, 191, 216),
    "plum": (221, 160, 221),
    "violet": (238, 130, 238),
    "magenta": (255, 0, 255),
    "orchid": (218, 112, 214),
    "medium violet red": (199, 21, 133),
    "pale violet red": (219, 112, 147),
    "deep pink": (255, 20, 147),
    "hot pink": (255, 105, 180),
    "light pink": (255, 182, 193),
    "pink": (255, 192, 203),
    "antique white": (250, 235, 215),
    "beige": (245, 245, 220),
    "bisque": (255, 228, 196),
    "blanched almond": (255, 235, 205),
    "wheat": (245, 222, 179),
    "corn silk": (255, 248, 220),
    "lemon chiffon": (255, 250, 205),
    "light golden rod yellow": (250, 250, 210),
    "light yellow": (255, 255, 224),
    "saddle brown": (139, 69, 19),
    "sienna": (160, 82, 45),
    "chocolate": (210, 105, 30),
    "peru": (205, 133, 63),
    "sandy brown": (244, 164, 96),
    "burly wood": (222, 184, 135),
    "tan": (210, 180, 140),
    "rosy brown": (188, 143, 143),
    "moccasin4B5": (255, 228, 181),
    "navajo white": (255, 222, 173),
    "peach puff": (255, 218, 185),
    "misty rose": (255, 228, 225),
    "lavender blush": (255, 240, 245),
    "linen": (250, 240, 230),
    "old lace": (253, 245, 230),
    "papaya whip": (255, 239, 213),
    "sea shell": (255, 245, 238),
    "mint cream": (245, 255, 250),
    "slate gray": (112, 128, 144),
    "light slate gray": (119, 136, 153),
    "light steel blue": (176, 196, 222),
    "lavender": (230, 230, 250),
    "floral white": (255, 250, 240),
    "alice blue": (240, 248, 255),
    "ghost white": (248, 248, 255),
    "honeydew": (240, 255, 240),
    "ivory": (255, 255, 240),
    "azure": (240, 255, 255),
    "snow": (255, 250, 250),
    "black": (0, 0, 0),
    "dim gray": (105, 105, 105),
    "gray": (128, 128, 128),
    "dark gray": (169, 169, 169),
    "silver": (192, 192, 192),
    "light gray": (211, 211, 211),
    "gainsboro": (220, 220, 220),
    "white smoke": (245, 245, 245),
    "white": (255, 255, 255),
}

scaledColorMap = {
    "maroon": (0.502, 0.0, 0.0),
    "dark red": (0.5451, 0.0, 0.0),
    "brown": (0.6471, 0.1647, 0.1647),
    "firebrick": (0.698, 0.1333, 0.1333),
    "crimson": (0.8627, 0.0784, 0.2353),
    "red": (1.0, 0.0, 0.0),
    "tomato": (1.0, 0.3882, 0.2784),
    "coral": (1.0, 0.498, 0.3137),
    "indian red": (0.8039, 0.3608, 0.3608),
    "light coral": (0.9412, 0.502, 0.502),
    "dark salmon": (0.9137, 0.5882, 0.4784),
    "salmon": (0.9804, 0.502, 0.4471),
    "light salmon": (1.0, 0.6275, 0.4784),
    "orange red": (1.0, 0.2706, 0.0),
    "dark orange": (1.0, 0.549, 0.0),
    "orange": (1.0, 0.6471, 0.0),
    "gold": (1.0, 0.8431, 0.0),
    "dark golden rod": (0.7216, 0.5255, 0.0431),
    "golden rod": (0.8549, 0.6471, 0.1255),
    "pale golden rod": (0.9333, 0.9098, 0.6667),
    "dark khaki": (0.7412, 0.7176, 0.4196),
    "khaki": (0.9412, 0.902, 0.549),
    "olive": (0.502, 0.502, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "yellow green": (0.6039, 0.8039, 0.1961),
    "dark olive green": (0.3333, 0.4196, 0.1843),
    "olive drab": (0.4196, 0.5569, 0.1373),
    "lawn green": (0.4863, 0.9882, 0.0),
    "chart reuse": (0.498, 1.0, 0.0),
    "green yellow": (0.6784, 1.0, 0.1843),
    "dark green": (0.0, 0.3922, 0.0),
    "green": (0.0, 0.502, 0.0),
    "forest green": (0.1333, 0.5451, 0.1333),
    "lime": (0.0, 1.0, 0.0),
    "lime green": (0.1961, 0.8039, 0.1961),
    "light green": (0.5647, 0.9333, 0.5647),
    "pale green": (0.5961, 0.9843, 0.5961),
    "dark sea green": (0.5608, 0.7373, 0.5608),
    "medium spring green": (0.0, 0.9804, 0.6039),
    "spring green": (0.0, 1.0, 0.498),
    "sea green": (0.1804, 0.5451, 0.3412),
    "medium aqua marine": (0.4, 0.8039, 0.6667),
    "medium sea green": (0.2353, 0.702, 0.4431),
    "light sea green": (0.1255, 0.698, 0.6667),
    "dark slate gray": (0.1843, 0.3098, 0.3098),
    "teal": (0.0, 0.502, 0.502),
    "dark cyan": (0.0, 0.5451, 0.5451),
    "aqua": (0.0, 1.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "light cyan": (0.8784, 1.0, 1.0),
    "dark turquoise": (0.0, 0.8078, 0.8196),
    "turquoise": (0.251, 0.8784, 0.8157),
    "medium turquoise": (0.2824, 0.8196, 0.8),
    "pale turquoise": (0.6863, 0.9333, 0.9333),
    "aqua marine": (0.498, 1.0, 0.8314),
    "powder blue": (0.6902, 0.8784, 0.902),
    "cadet blue": (0.3725, 0.6196, 0.6275),
    "steel blue": (0.2745, 0.5098, 0.7059),
    "corn flower blue": (0.3922, 0.5843, 0.9294),
    "deep sky blue": (0.0, 0.749, 1.0),
    "dodger blue": (0.1176, 0.5647, 1.0),
    "light blue": (0.6784, 0.8471, 0.902),
    "sky blue": (0.5294, 0.8078, 0.9216),
    "light sky blue": (0.5294, 0.8078, 0.9804),
    "midnight blue": (0.098, 0.098, 0.4392),
    "navy": (0.0, 0.0, 0.502),
    "dark blue": (0.0, 0.0, 0.5451),
    "medium blue": (0.0, 0.0, 0.8039),
    "blue": (0.0, 0.0, 1.0),
    "royal blue": (0.2549, 0.4118, 0.8824),
    "blue violet": (0.5412, 0.1686, 0.8863),
    "indigo": (0.2941, 0.0, 0.5098),
    "dark slate blue": (0.2824, 0.2392, 0.5451),
    "slate blue": (0.4157, 0.3529, 0.8039),
    "medium slate blue": (0.4824, 0.4078, 0.9333),
    "medium purple": (0.5765, 0.4392, 0.8588),
    "dark magenta": (0.5451, 0.0, 0.5451),
    "dark violet": (0.5804, 0.0, 0.8275),
    "dark orchid": (0.6, 0.1961, 0.8),
    "medium orchid": (0.7294, 0.3333, 0.8275),
    "purple": (0.502, 0.0, 0.502),
    "thistle": (0.8471, 0.749, 0.8471),
    "plum": (0.8667, 0.6275, 0.8667),
    "violet": (0.9333, 0.5098, 0.9333),
    "magenta": (1.0, 0.0, 1.0),
    "orchid": (0.8549, 0.4392, 0.8392),
    "medium violet red": (0.7804, 0.0824, 0.5216),
    "pale violet red": (0.8588, 0.4392, 0.5765),
    "deep pink": (1.0, 0.0784, 0.5765),
    "hot pink": (1.0, 0.4118, 0.7059),
    "light pink": (1.0, 0.7137, 0.7569),
    "pink": (1.0, 0.7529, 0.7961),
    "antique white": (0.9804, 0.9216, 0.8431),
    "beige": (0.9608, 0.9608, 0.8627),
    "bisque": (1.0, 0.8941, 0.7686),
    "blanched almond": (1.0, 0.9216, 0.8039),
    "wheat": (0.9608, 0.8706, 0.702),
    "corn silk": (1.0, 0.9725, 0.8627),
    "lemon chiffon": (1.0, 0.9804, 0.8039),
    "light golden rod yellow": (0.9804, 0.9804, 0.8235),
    "light yellow": (1.0, 1.0, 0.8784),
    "saddle brown": (0.5451, 0.2706, 0.0745),
    "sienna": (0.6275, 0.3216, 0.1765),
    "chocolate": (0.8235, 0.4118, 0.1176),
    "peru": (0.8039, 0.5216, 0.2471),
    "sandy brown": (0.9569, 0.6431, 0.3765),
    "burly wood": (0.8706, 0.7216, 0.5294),
    "tan": (0.8235, 0.7059, 0.549),
    "rosy brown": (0.7373, 0.5608, 0.5608),
    "moccasin4B5": (1.0, 0.8941, 0.7098),
    "navajo white": (1.0, 0.8706, 0.6784),
    "peach puff": (1.0, 0.8549, 0.7255),
    "misty rose": (1.0, 0.8941, 0.8824),
    "lavender blush": (1.0, 0.9412, 0.9608),
    "linen": (0.9804, 0.9412, 0.902),
    "old lace": (0.9922, 0.9608, 0.902),
    "papaya whip": (1.0, 0.9373, 0.8353),
    "sea shell": (1.0, 0.9608, 0.9333),
    "mint cream": (0.9608, 1.0, 0.9804),
    "slate gray": (0.4392, 0.502, 0.5647),
    "light slate gray": (0.4667, 0.5333, 0.6),
    "light steel blue": (0.6902, 0.7686, 0.8706),
    "lavender": (0.902, 0.902, 0.9804),
    "floral white": (1.0, 0.9804, 0.9412),
    "alice blue": (0.9412, 0.9725, 1.0),
    "ghost white": (0.9725, 0.9725, 1.0),
    "honeydew": (0.9412, 1.0, 0.9412),
    "ivory": (1.0, 1.0, 0.9412),
    "azure": (0.9412, 1.0, 1.0),
    "snow": (1.0, 0.9804, 0.9804),
    "black": (0.0, 0.0, 0.0),
    "dim gray": (0.4118, 0.4118, 0.4118),
    "gray": (0.502, 0.502, 0.502),
    "dark gray": (0.6627, 0.6627, 0.6627),
    "silver": (0.7529, 0.7529, 0.7529),
    "light gray": (0.8275, 0.8275, 0.8275),
    "gainsboro": (0.8627, 0.8627, 0.8627),
    "white smoke": (0.9608, 0.9608, 0.9608),
    "white": (1.0, 1.0, 1.0),
}

with open("RGBcolors.json", "w") as f:
    json.dump(colorMap, f)