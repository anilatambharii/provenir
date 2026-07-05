"""Generate the 1280x640 social preview image for provenir's GitHub repository."""

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640

# Color palette
BG          = (13,  17,  23)    # #0D1117  GitHub dark
CARD_BG     = (22,  27,  34)    # #161B22
BORDER      = (48,  54,  61)    # #30363D
WHITE       = (255, 255, 255)
GRAY        = (139, 148, 158)   # #8B949E
BLUE        = (47,  129, 247)   # #2F81F7
PURPLE      = (124, 58,  237)   # #7C3AED
GREEN       = (46,  160, 67)    # #2EA043
GREEN_TEXT  = (63,  185, 80)    # #3FB950
ORANGE      = (240, 136, 62)    # #F0883E
AMBER       = (210, 153, 34)    # #D29922
BLUE_TEXT   = (88,  166, 255)   # #58A6FF

FONTS = "C:/Windows/Fonts"

def font(name, size):
    try:
        return ImageFont.truetype(f"{FONTS}/{name}", size)
    except Exception:
        return ImageFont.load_default()

f_logo    = font("arialbd.ttf", 104)
f_tag     = font("arial.ttf",   34)
f_sub     = font("arial.ttf",   22)
f_badge   = font("arialbd.ttf", 17)
f_card_h  = font("arialbd.ttf", 20)
f_card_b  = font("arial.ttf",   15)
f_pip     = font("cour.ttf",    24)
f_feat    = font("arial.ttf",   19)
f_feat_b  = font("arialbd.ttf", 19)

img  = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

# ── subtle left gradient strip ─────────────────────────────────────────────
for x in range(0, 6):
    alpha = int(80 * (1 - x / 6))
    draw.line([(x, 0), (x, H)], fill=(124, 58, 237, alpha))

# ══ LEFT PANEL  (x=80 … 600) ══════════════════════════════════════════════

LX = 80

# Logo
draw.text((LX, 64), "provenir", fill=WHITE, font=f_logo)
bb = draw.textbbox((LX, 64), "provenir", font=f_logo)
logo_w = bb[2] - bb[0]
logo_b = bb[3]

# Purple accent underline
draw.rectangle([LX, logo_b + 6, LX + logo_w, logo_b + 12], fill=PURPLE)

# Tagline (two lines for space)
draw.text((LX, logo_b + 28), "The Trust Layer for", fill=GRAY, font=f_tag)
draw.text((LX, logo_b + 68), "Model Post-Training", fill=WHITE, font=f_tag)

# Badges row
def badge(x, y, text, bg, fg, border_col):
    bb = draw.textbbox((0, 0), text, font=f_badge)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    pad_x, pad_y = 14, 6
    bw = tw + 2 * pad_x
    bh = th + 2 * pad_y
    draw.rounded_rectangle([x, y, x + bw, y + bh], radius=6,
                            fill=bg, outline=border_col, width=1)
    draw.text((x + pad_x, y + pad_y - bb[1]), text, fill=fg, font=f_badge)
    return x + bw + 10

badge_y = logo_b + 124
bx = LX
bx = badge(bx, badge_y, "v0.5.1",          CARD_BG, BLUE_TEXT, BORDER)
bx = badge(bx, badge_y, "1,153 tests ✓",   (21, 43, 29), GREEN_TEXT, (35, 134, 54))
bx = badge(bx, badge_y, "Apache-2.0",       CARD_BG, GRAY, BORDER)

# Separator line
SEP_X = 620
draw.line([(SEP_X, 40), (SEP_X, H - 40)], fill=BORDER, width=1)

# ══ RIGHT PANEL — feature list  (x=650 … 1200) ════════════════════════════

RX = 660
RY = 52

features = [
    (BLUE,   "RL Flight Recorder",       "KL · entropy · gradient anomalies · per-step"),
    (PURPLE, "Reward-Hacking Detection",  "Length, format, test-tampering & verifier gaming"),
    (GREEN_TEXT, "Contamination Firewall","13-gram + embedding train/eval overlap guard"),
    (ORANGE, "Signed Model Passport",     "HMAC-SHA256 BOM · EU AI Act Art. 12 ready"),
    (AMBER,  "Loop Doctor",               "Differential diagnosis: eval · reward · algo · data"),
    (BLUE_TEXT,"Agentic Environments",    "Multi-turn tool-use with verifiable rewards"),
]

row_h = 76
for i, (color, title, body) in enumerate(features):
    ry = RY + i * row_h
    # Dot
    cx, cy = RX + 10, ry + 16
    draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=color)
    # Title
    draw.text((RX + 28, ry + 4), title, fill=WHITE, font=f_feat_b)
    # Body
    draw.text((RX + 28, ry + 28), body, fill=GRAY, font=f_feat)

# ══ BOTTOM STRIP ══════════════════════════════════════════════════════════

strip_y = H - 78
draw.rectangle([0, strip_y, W, H], fill=CARD_BG)
draw.line([(0, strip_y), (W, strip_y)], fill=BORDER, width=1)

# pip install command centered
pip_txt = "pip install provenir"
pb = draw.textbbox((0, 0), pip_txt, font=f_pip)
pw = pb[2] - pb[0]
pip_x = (W - pw) // 2 - 50
pip_y = strip_y + 18
# box around pip command
draw.rounded_rectangle([pip_x - 18, pip_y - 8, pip_x + pw + 18, pip_y + 38],
                        radius=6, fill=BG, outline=BORDER, width=1)
draw.text((pip_x, pip_y), pip_txt, fill=GREEN_TEXT, font=f_pip)

# "Get started:" label
label = "Get started:"
lb = draw.textbbox((0, 0), label, font=f_sub)
draw.text((pip_x - lb[2] - lb[0] - 28, pip_y + 2), label, fill=GRAY, font=f_sub)

# Right side of strip: website
site_txt = "github.com/anilatambharii/provenir"
sb = draw.textbbox((0, 0), site_txt, font=f_sub)
draw.text((W - sb[2] - 36, strip_y + 22), site_txt, fill=GRAY, font=f_sub)

out = "assets/social_preview.png"
img.save(out, "PNG")
print(f"Saved {W}x{H} to {out}")
