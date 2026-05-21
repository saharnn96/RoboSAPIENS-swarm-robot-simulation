from PIL import Image
import numpy as np

SIZE = 500        # pixels
SCALE = 50        # pixels per world unit (world is 10x10)
T = 0.25          # wall thickness in world units

img = np.ones((SIZE, SIZE), dtype=np.uint8) * 255  # white = free space

def px(v):
    return int(v * SCALE)

def fill(x, y, w, h):
    """Fill rectangle (world units, y=0 at bottom) with black."""
    r0 = SIZE - px(y + h)
    r1 = SIZE - px(y)
    c0 = px(x)
    c1 = px(x + w)
    img[max(0,r0):min(SIZE,r1), max(0,c0):min(SIZE,c1)] = 0

def hwall(x0, x1, y, door=None, door_w=1.2):
    """Horizontal wall from x0 to x1 at height y, with optional door gap."""
    if door is None:
        fill(x0, y, x1 - x0, T)
    else:
        fill(x0, y, door - x0, T)
        fill(door + door_w, y, x1 - (door + door_w), T)

def vwall(x, y0, y1, door=None, door_w=1.2):
    """Vertical wall from y0 to y1 at x, with optional door gap."""
    if door is None:
        fill(x, y0, T, y1 - y0)
    else:
        fill(x, y0, T, door - y0)
        fill(x, door + door_w, T, y1 - (door + door_w))

# ── Outer border ──────────────────────────────────────────────────────────────
hwall(0, 10, 0)          # bottom
hwall(0, 10, 10 - T)     # top
vwall(0, 0, 10)          # left
vwall(10 - T, 0, 10)     # right

# ── Room layout (4 rooms + central cross-corridor) ────────────────────────────
#
#   +----------+     +----------+
#   |  Room A  |     |  Room B  |
#   |          |     |          |
#   +----++----+     +----++----+
#        ||  cross corridor  ||
#   +----++----+     +----++----+
#   |  Room C  |     |  Room D  |
#   |          |     |          |
#   +----------+     +----------+
#
# Dividing lines: x=4.3, x=5.7,  y=4.3, y=5.7

MID_X1, MID_X2 = 4.3, 5.7   # corridor x span
MID_Y1, MID_Y2 = 4.3, 5.7   # corridor y span

# Horizontal dividers (top rooms / corridor / bottom rooms)
hwall(T, MID_X1, MID_Y2, door=1.8, door_w=1.2)        # bottom of Room A
hwall(MID_X2, 10-T, MID_Y2, door=7.0, door_w=1.2)     # bottom of Room B
hwall(T, MID_X1, MID_Y1, door=1.8, door_w=1.2)        # top of Room C
hwall(MID_X2, 10-T, MID_Y1, door=7.0, door_w=1.2)     # top of Room D

# Vertical dividers (left rooms / corridor / right rooms)
vwall(MID_X1, MID_Y2, 10-T, door=7.5, door_w=1.0)     # right of Room A
vwall(MID_X2, MID_Y2, 10-T, door=7.5, door_w=1.0)     # left of Room B
vwall(MID_X1, T, MID_Y1, door=1.5, door_w=1.0)        # right of Room C
vwall(MID_X2, T, MID_Y1, door=1.5, door_w=1.0)        # left of Room D

# ── Internal detail walls ─────────────────────────────────────────────────────
# Room A — partial divider with gap
hwall(T, 2.8, 8.2, door=1.2, door_w=0.9)

# Room B — alcove wall
vwall(7.8, MID_Y2, 9.0)

# Room C — L-shaped obstacle
hwall(T, 2.5, 2.8)
vwall(2.5, 2.8, 3.8)

# Room D — short wall
hwall(6.5, 9.0, 2.5, door=7.5, door_w=0.9)

# Corridor junction pillars (small square pillars at cross-center)
fill(4.8, 4.8, 0.4, 0.4)

Image.fromarray(img, mode='L').save('robot_map.png')
print("Complex map saved as robot_map.png")
