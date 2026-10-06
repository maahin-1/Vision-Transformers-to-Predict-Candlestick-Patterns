"""Draw the last 20 candles the way the training screenshots look (Plotly default theme).

The ViT was trained on 700x500 downscaled screenshots of a Plotly chart showing 20 candles.
Rendering the same picture straight from OHLC data means the model never depends on where a
browser window sits on screen. Geometry below was measured from the training images.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 700, 500
PLOT_L, PLOT_T, PLOT_R, PLOT_B = 76, 106, 624, 322
N_CANDLES = 20
SS = 4  # supersampling factor, mimics the downscale of the original 4K screenshots

BG = (223, 231, 244)
GRID = (237, 242, 249)
UP = (61, 153, 112)
DOWN = (255, 65, 54)
TEXT = (42, 63, 95)
Y_PAD = 0.056
LINE_PX = 1.3


def _blend(color, alpha=0.5, base=BG):
    return tuple(int(round(c * alpha + b * (1 - alpha))) for c, b in zip(color, base))


def _nice_step(span, plot_px, target_px=40):
    raw = span / max(plot_px / target_px, 1)
    mag = 10 ** np.floor(np.log10(raw))
    best = min((m * mag for m in (1, 2, 5, 10)), key=lambda s: abs(np.log(s / raw)))
    return best


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def render_chart(opens, highs, lows, closes, times=None):
    """Return a (500, 700, 3) uint8 RGB image of the last 20 candles."""
    o, h, l, c = (np.asarray(x, dtype=float)[-N_CANDLES:] for x in (opens, highs, lows, closes))
    n = len(o)
    if n < N_CANDLES:
        raise ValueError(f'need {N_CANDLES} candles, got {n}')

    img = Image.new('RGB', (WIDTH * SS, HEIGHT * SS), (255, 255, 255))
    d = ImageDraw.Draw(img)
    pl, pt, pr, pb = (v * SS for v in (PLOT_L, PLOT_T, PLOT_R, PLOT_B))
    d.rectangle([pl, pt, pr, pb], fill=BG)

    lo, hi = float(l.min()), float(h.max())
    span = hi - lo or max(abs(hi) * 1e-3, 1e-6)
    y_lo, y_hi = lo - Y_PAD * span, hi + Y_PAD * span

    def y_px(v):
        return pb - (v - y_lo) / (y_hi - y_lo) * (pb - pt)

    spacing = (pr - pl) / N_CANDLES
    cx = [pl + (i + 0.5) * spacing for i in range(n)]

    # horizontal grid
    step = _nice_step(y_hi - y_lo, (PLOT_B - PLOT_T))
    tick = np.ceil(y_lo / step) * step
    while tick <= y_hi:
        y = y_px(tick)
        d.line([pl, y, pr, y], fill=GRID, width=SS)
        tick += step
    # vertical grid every 5 minutes
    small = _font(7 * SS)
    if times is not None:
        for i, t in enumerate(list(times)[-n:]):
            if t.minute % 5 == 0:
                d.line([cx[i], pt, cx[i], pb], fill=GRID, width=SS)
                label = t.strftime('%H:%M')
                tw = d.textlength(label, font=small)
                d.text((cx[i] - tw / 2, pb + 2 * SS), label, fill=TEXT, font=small)

    body_w = spacing * 0.5
    lw = max(1, int(round(LINE_PX * SS)))
    for i in range(n):
        color = UP if c[i] >= o[i] else DOWN
        top, bot = y_px(max(o[i], c[i])), y_px(min(o[i], c[i]))
        x = cx[i]
        d.line([x, y_px(h[i]), x, top], fill=color, width=lw)
        d.line([x, bot, x, y_px(l[i])], fill=color, width=lw)
        if bot - top < lw:  # doji: flat bar
            d.line([x - body_w / 2, (top + bot) / 2, x + body_w / 2, (top + bot) / 2], fill=color, width=lw)
        else:
            d.rectangle([x - body_w / 2, top, x + body_w / 2, bot], fill=_blend(color), outline=color, width=lw)

    return np.asarray(img.resize((WIDTH, HEIGHT), Image.Resampling.BOX))
