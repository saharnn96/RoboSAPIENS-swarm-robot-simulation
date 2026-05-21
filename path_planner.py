import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.ndimage import binary_dilation

MAP_FILE      = 'robot_map.png'
WORLD_SIZE    = 10.0
GRID_SIZE     = 200
ROBOT_RADIUS  = 0.15
BLOCK_SIZE    = 0.5
MIN_AB_DIST   = 3.5   # min distance between A and B within a pair
MIN_PT_DIST   = 1.5   # min distance between any two points across pairs

# Per-pair colour: (A colour, B colour, path colour)
PAIR_STYLES = [
    ('#e74c3c', '#c0392b', '#e74c3c'),   # pair 1 — red
    ('#3498db', '#1a5276', '#3498db'),   # pair 2 — blue
    ('#2ecc71', '#1e8449', '#2ecc71'),   # pair 3 — green
]


# ── Map loading ───────────────────────────────────────────────────────────────

def load_map():
    img = Image.open(MAP_FILE).convert('L')
    raw = np.array(img)
    small = np.array(img.resize((GRID_SIZE, GRID_SIZE), Image.LANCZOS))
    occupied = (small < 128).astype(np.uint8)
    rc = int(ROBOT_RADIUS * GRID_SIZE / WORLD_SIZE)
    if rc > 0:
        occupied = binary_dilation(
            occupied, structure=np.ones((2*rc+1, 2*rc+1))
        ).astype(np.uint8)
    return raw, occupied


# ── Coordinate helpers ────────────────────────────────────────────────────────

def w2g(wx, wy):
    c = int(wx * GRID_SIZE / WORLD_SIZE)
    r = int((WORLD_SIZE - wy) * GRID_SIZE / WORLD_SIZE)
    return int(np.clip(r, 0, GRID_SIZE-1)), int(np.clip(c, 0, GRID_SIZE-1))

def g2w(r, c):
    return (c+0.5)*WORLD_SIZE/GRID_SIZE, WORLD_SIZE-(r+0.5)*WORLD_SIZE/GRID_SIZE


# ── A* ────────────────────────────────────────────────────────────────────────

_MOVES = [(-1,-1,1.414),(-1,0,1.0),(-1,1,1.414),
          ( 0,-1,1.0),            ( 0,1,1.0),
          ( 1,-1,1.414),( 1,0,1.0),( 1,1,1.414)]

def astar(grid, start, goal):
    def h(a, b): return np.hypot(a[0]-b[0], a[1]-b[1])
    open_set = [(h(start,goal), 0.0, start)]
    g = {start: 0.0}; prev = {}
    rows, cols = grid.shape
    while open_set:
        _, gc, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in prev: path.append(cur); cur = prev[cur]
            return [start] + path[::-1]
        if gc > g.get(cur, float('inf')): continue
        for dr, dc, cost in _MOVES:
            nr, nc = cur[0]+dr, cur[1]+dc
            if not (0<=nr<rows and 0<=nc<cols) or grid[nr,nc]: continue
            ng = gc + cost; nb = (nr,nc)
            if ng < g.get(nb, float('inf')):
                g[nb]=ng; prev[nb]=cur
                heapq.heappush(open_set, (ng+h(nb,goal), ng, nb))
    return None


# ── PathPlanner ───────────────────────────────────────────────────────────────

class PathPlanner:
    def __init__(self):
        self.raw_img, self.base_grid = load_map()
        self.working_grid = self.base_grid.copy()

        self._path_artists     = []
        self._block_artists    = []
        self._endpoint_artists = []
        self.pairs = []   # list of (a_pt, b_pt) per pair

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(self.raw_img, cmap='gray',
                       extent=[0, WORLD_SIZE, 0, WORLD_SIZE], origin='upper')
        self.ax.set_xlim(0, WORLD_SIZE)
        self.ax.set_ylim(0, WORLD_SIZE)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',   self._on_key)
        self.fig.text(0.5, 0.01,
                      'Click → place obstacle & reroute all   |   R → reset',
                      ha='center', fontsize=9, color='gray')

        self._randomize_endpoints()
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()

    # ── random point selection ────────────────────────────────────────────────

    def _random_free_point(self, avoid_pts, min_dist):
        rng = np.random.default_rng()
        for _ in range(20_000):
            wx = rng.uniform(0.5, WORLD_SIZE - 0.5)
            wy = rng.uniform(0.5, WORLD_SIZE - 0.5)
            if self.base_grid[w2g(wx, wy)]:
                continue
            if any(np.hypot(wx-ax, wy-ay) < min_dist for ax, ay in avoid_pts):
                continue
            return wx, wy
        raise RuntimeError("Could not place a point — map too cluttered.")

    def _randomize_endpoints(self):
        self.pairs.clear()
        all_pts = []
        for _ in range(3):
            a = self._random_free_point(all_pts, MIN_PT_DIST)
            all_pts.append(a)
            b = self._random_free_point(
                all_pts + [a],
                max(MIN_AB_DIST, MIN_PT_DIST)
            )
            # Ensure B is far enough from A specifically
            while np.hypot(a[0]-b[0], a[1]-b[1]) < MIN_AB_DIST:
                b = self._random_free_point(all_pts + [a], MIN_PT_DIST)
            all_pts.append(b)
            self.pairs.append((a, b))

        self._draw_endpoints()
        self._reroute_all()

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_endpoints(self):
        for a in self._endpoint_artists:
            a.remove()
        self._endpoint_artists.clear()

        for i, ((ax, ay), (bx, by)) in enumerate(self.pairs):
            ca, cb, _ = PAIR_STYLES[i]
            n = i + 1

            for wx, wy, color, label, marker in [
                (ax, ay, ca, f'A{n}', 'o'),
                (bx, by, cb, f'B{n}', 's'),
            ]:
                mk, = self.ax.plot(wx, wy, marker, color=color, markersize=13,
                                   zorder=8, markeredgecolor='white',
                                   markeredgewidth=1.8)
                tx = self.ax.text(wx+0.15, wy+0.15, label, color=color,
                                  fontsize=11, fontweight='bold', zorder=9)
                self._endpoint_artists.extend([mk, tx])

    def _clear_paths(self):
        for a in self._path_artists:
            a.remove()
        self._path_artists.clear()

    def _reroute_all(self):
        self._clear_paths()
        lengths = []
        for i, (a_pt, b_pt) in enumerate(self.pairs):
            _, _, path_color = PAIR_STYLES[i]
            path = astar(self.working_grid, w2g(*a_pt), w2g(*b_pt))
            if path:
                wxs = [g2w(r,c)[0] for r,c in path]
                wys = [g2w(r,c)[1] for r,c in path]
                line, = self.ax.plot(wxs, wys, '-', color=path_color,
                                     linewidth=2.5, zorder=4, alpha=0.8)
                step = max(1, len(path) // 12)
                dots, = self.ax.plot(wxs[::step], wys[::step], 's',
                                     color=path_color, markersize=6,
                                     zorder=5, alpha=0.85)
                self._path_artists.extend([line, dots])
                length = sum(np.hypot(wxs[j]-wxs[j-1], wys[j]-wys[j-1])
                             for j in range(1, len(wxs)))
                lengths.append(f'P{i+1}: {length:.1f}m')
            else:
                lengths.append(f'P{i+1}: blocked')

        self.ax.set_title('  |  '.join(lengths), fontsize=10)
        self.fig.canvas.draw()

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        wx, wy = event.xdata, event.ydata
        half = BLOCK_SIZE / 2

        r0, c0 = w2g(wx-half, wy+half)
        r1, c1 = w2g(wx+half, wy-half)
        r0, r1 = min(r0,r1), max(r0,r1)+1
        c0, c1 = min(c0,c1), max(c0,c1)+1
        self.working_grid[r0:r1, c0:c1] = 1

        rect = mpatches.FancyBboxPatch(
            (wx-half, wy-half), BLOCK_SIZE, BLOCK_SIZE,
            boxstyle='square,pad=0', facecolor='#2c3e50',
            edgecolor='#7f8c8d', linewidth=1, zorder=6
        )
        self.ax.add_patch(rect)
        self._block_artists.append(rect)
        self._reroute_all()

    def _on_key(self, event):
        if event.key.lower() != 'r':
            return
        for a in self._block_artists:
            a.remove()
        self._block_artists.clear()
        self.working_grid = self.base_grid.copy()
        self._randomize_endpoints()


if __name__ == '__main__':
    PathPlanner()
