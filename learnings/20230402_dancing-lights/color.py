#!/usr/bin/env python3

# https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors - good read
# https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
# from https://stackoverflow.com/a/13781114/6561141

from typing import Iterable, Tuple
import colorsys
import itertools
from fractions import Fraction
from pprint import pprint

def zenos_dichotomy() -> Iterable[Fraction]:
    """
    http://en.wikipedia.org/wiki/1/2_%2B_1/4_%2B_1/8_%2B_1/16_%2B_%C2%B7_%C2%B7_%C2%B7
    """
    for k in itertools.count():
        yield Fraction(1,2**k)

def fracs() -> Iterable[Fraction]:
    """
    [Fraction(0, 1), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 16), Fraction(3, 16), ...]
    [0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, ...]
    """
    yield Fraction(0)
    for k in zenos_dichotomy():
        i = k.denominator # [1,2,4,8,16,...]
        for j in range(1,i,2):
            yield Fraction(j,i)

# can be used for the v in hsv to map linear values 0..1 to something that looks equidistant
# bias = lambda x: (math.sqrt(x/3)/Fraction(2,3)+Fraction(1,3))/Fraction(6,5)

HSVTuple = Tuple[Fraction, Fraction, Fraction]
RGBTuple = Tuple[float, float, float]

def hue_to_tones(h: Fraction) -> Iterable[HSVTuple]:
    for s in [Fraction(6,10)]: # optionally use range
        for v in [Fraction(8,10),Fraction(5,10)]: # could use range too
            yield (h, s, v) # use bias for v here if you use range

def hsv_to_rgb(x: HSVTuple) -> RGBTuple:
    return colorsys.hsv_to_rgb(*map(float, x))

flatten = itertools.chain.from_iterable

def hsvs() -> Iterable[HSVTuple]:
    return flatten(map(hue_to_tones, fracs()))

def rgbs() -> Iterable[RGBTuple]:
    return map(hsv_to_rgb, hsvs())

def rgb_to_css(x: RGBTuple) -> str:
    uint8tuple = map(lambda y: int(y*255), x)
    return "rgb({},{},{})".format(*uint8tuple)

def get_rgb_colors(x: RGBTuple) -> str:
    uint8tuple = map(lambda y: int(y*255), x)
    return tuple(uint8tuple)


def css_colors() -> Iterable[str]:
    return map(rgb_to_css, rgbs())

def rgb_colors() -> Iterable[str]:
    return map(get_rgb_colors, rgbs())

def get_color_samples(num_color_samples): # https://stackoverflow.com/a/29643643/6561141
    return list(itertools.islice(rgb_colors(), num_color_samples))


def convert_hex2rgb(hexcode):
    hexcode = hexcode.lstrip('#')
    rgb = tuple(int(hexcode[i:i+2], 16) for i in (0, 2, 4)) # https://stackoverflow.com/a/29643643/6561141
    return rgb 

# Kelly
#FFB300
#803E75
#FF6800
#A6BDD7
#C10020
#CEA262
#817066

# Kelly - The following will not be good for people with defective color vision
#007D34
#F6768E
#00538A
#FF7A5C
#53377A
#FF8E00
#B32851
#F4C800
#7F180D
#93AA00
#593315
#F13A13
#232C16

# Boynton Optimized
#0000FF
#FF0000 
#00FF00
#FFFF00
#FF00FF
#FF8080 
#808080 
#800000
#FF8000

static_colors = ["#FFB300", "#803E75", "#FF6800", "#A6BDD7", "#C10020", "#CEA262", "#817066", "#007D34", "#F6768E", "#00538A", "#FF7A5C", "#53377A", "#FF8E00", "#B32851", "#F4C800", "#7F180D", "#93AA00", "#593315", "#F13A13", "#232C16", "#0000FF", "#FF0000", "#00FF00", "#FFFF00", "#FF00FF", "#FF8080", "#808080", "#800000", "#FF8000"]
# static_colors = ["#232C16", "#0000FF", "#FF0000", "#00FF00", "#FFFF00", "#FF00FF", "#FF8080", "#808080", "#800000", "#FF8000"]
# static_colors = ["#00FF00", "#FFFF00", "#FF00FF", "#FF8080", "#808080", "#800000"]
static_colors = list(map(convert_hex2rgb, static_colors))

def get_static_color_samples(num_color_samples):
    assert num_color_samples <= len(static_colors)
    return static_colors[:num_color_samples]

if __name__ == "__main__":
    # sample 100 colors in css format
    # sample_colors = list(itertools.islice(css_colors(), 100))
    # sample_colors = get_color_samples(num_color_samples=10)
    sample_colors = get_static_color_samples(num_color_samples=10)
    pprint(sample_colors)
