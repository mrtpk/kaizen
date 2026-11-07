"""Number-line helper for the quantization-basics figures (built on svglib)."""

from svglib import GRAY_MID, INK, MUTE, _c


class NumberLine:
    """A horizontal number line from value `vmin` at x0 to value `vmax` at x1."""

    def __init__(self, d, x0, x1, y, vmin, vmax, ticks=(), tick_fmt="{:g}", name=None, col=GRAY_MID):
        self.d, self.x0, self.x1, self.y, self.vmin, self.vmax = d, x0, x1, y, vmin, vmax
        d.line(x0, y, x1, y, col, sw=2)
        for t in ticks:
            x = self.px(t)
            d.line(x, y - 6, x, y + 6, col, sw=1.4)
            d.text(x, y + 24, tick_fmt.format(t), 12, MUTE, anchor="middle", mono=True)
        if name:
            d.text(x0 - 14, y + 5, name, 13, INK, "bold", anchor="end")

    def px(self, v):
        return self.x0 + (v - self.vmin) / (self.vmax - self.vmin) * (self.x1 - self.x0)

    def point(self, v, label, col, above=True, r=6, size=12.5):
        x = self.px(v)
        self.d.circle(x, self.y, r, _c(col))
        ly = self.y - 16 if above else self.y + 44
        self.d.text(x, ly, label, size, col, "bold", anchor="middle", mono=True)
        return x
