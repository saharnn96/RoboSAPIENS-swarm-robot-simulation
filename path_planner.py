import heapq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation

MAP_FILE    = 'robot_map.png'
WORLD_SIZE  = 10.0
GRID_SIZE   = 200          # pathfinding resolution (200x200 cells)
ROBOT_RADIUS = 0.15        # world units — inflates walls so path keeps clearance


# ── Map loading ───────────────────────────────────────────────────────────────

def load_map():
    img = Image.open(MAP_FILE).convert('L')
    raw = np.array(img)                             # 500x500, 0=wall, 255=free
    small = np.array(img.resize((GRID_SIZE, GRID_SIZE), Image.LANCZOS))
    occupied = (small < 128).astype(np.uint8)       # 1=wall, 0=free

    radius_cells = int(ROBOT_RADIUS * GRID_SIZE / WORLD_SIZE)
    if radius_cells > 0:
        struct = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1))
        inflated = binary_dilation(occupied, structure=struct).astype(np.uint8)
    else:
        inflated = occupied

    return raw, inflated


# ── Coordinate conversion ─────────────────────────────────────────────────────

def world_to_grid(wx, wy):
    c = int(wx * GRID_SIZE / WORLD_SIZE)
    r = int((WORLD_SIZE - wy) * GRID_SIZE / WORLD_SIZE)
    return np.clip(r, 0, GRID_SIZE - 1), np.clip(c, 0, GRID_SIZE - 1)


def grid_to_world(r, c):
    wx = (c + 0.5) * WORLD_SIZE / GRID_SIZE
    wy = WORLD_SIZE - (r + 0.5) * WORLD_SIZE / GRID_SIZE
    return wx, wy


# ── A* pathfinding ────────────────────────────────────────────────────────────

MOVES_8 = [(-1,-1,1.414),(-1,0,1.0),(-1,1,1.414),
           ( 0,-1,1.0),          ( 0,1,1.0),
           ( 1,-1,1.414),( 1,0,1.0),( 1,1,1.414)]

def astar(grid, start, goal):
    rows, cols = grid.shape

    def h(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    open_set = [(h(start, goal), 0.0, start)]
    g = {start: 0.0}
    prev = {}

    while open_set:
        _, g_cur, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in prev:
                path.append(cur)
                cur = prev[cur]
            path.append(start)
            return path[::-1]

        if g_cur > g.get(cur, float('inf')):
            continue

        for dr, dc, cost in MOVES_8:
            nr, nc = cur[0]+dr, cur[1]+dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc]:
                continue
            ng = g_cur + cost
            nb = (nr, nc)
            if ng < g.get(nb, float('inf')):
                g[nb] = ng
                prev[nb] = cur
                heapq.heappush(open_set, (ng + h(nb, goal), ng, nb))

    return None


# ── Interactive planner ───────────────────────────────────────────────────────

class PathPlanner:
    def __init__(self):
        self.raw_img, self.grid = load_map()
        self.clicks = []
        self._artists = []

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(self.raw_img, cmap='gray',
                       extent=[0, WORLD_SIZE, 0, WORLD_SIZE], origin='upper')
        self.ax.set_xlim(0, WORLD_SIZE)
        self.ax.set_ylim(0, WORLD_SIZE)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self._set_title('idle')

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        plt.tight_layout()
        plt.show()

    def _set_title(self, state):
        msgs = {
            'idle':    'Click Point A to start',
            'wait_b':  'Point A set — now click Point B',
            'found':   'Path found! Click anywhere to reset',
            'wall_a':  'Point A is inside a wall — click elsewhere',
            'wall_b':  'Point B is inside a wall — click elsewhere',
            'no_path': 'No path found — try different points. Click to reset',
        }
        self.ax.set_title(msgs.get(state, state), fontsize=11)

    def _clear(self):
        for a in self._artists:
            a.remove()
        self._artists.clear()
        self.clicks.clear()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        wx, wy = event.xdata, event.ydata

        if len(self.clicks) >= 2:
            self._clear()
            self._set_title('idle')
            self.fig.canvas.draw()
            return

        r, c = world_to_grid(wx, wy)
        if self.grid[r, c]:
            self._set_title('wall_a' if not self.clicks else 'wall_b')
            self.fig.canvas.draw()
            return

        self.clicks.append((wx, wy))
        label, color = ('A', '#2ecc71') if len(self.clicks) == 1 else ('B', '#e74c3c')

        mk, = self.ax.plot(wx, wy, 'o', color=color, markersize=12,
                           zorder=6, markeredgecolor='white', markeredgewidth=1.5)
        tx = self.ax.text(wx + 0.15, wy + 0.15, label, color=color,
                          fontsize=13, fontweight='bold', zorder=7)
        self._artists.extend([mk, tx])

        if len(self.clicks) == 1:
            self._set_title('wait_b')
        else:
            self._plan()

        self.fig.canvas.draw()

    def _plan(self):
        (ax, ay), (bx, by) = self.clicks
        start = world_to_grid(ax, ay)
        goal  = world_to_grid(bx, by)

        self.ax.set_title('Computing…', fontsize=11)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        path = astar(self.grid, start, goal)

        if path is None:
            self._set_title('no_path')
            return

        wxs = [grid_to_world(r, c)[0] for r, c in path]
        wys = [grid_to_world(r, c)[1] for r, c in path]

        line, = self.ax.plot(wxs, wys, '-', color='#3498db',
                             linewidth=2.5, zorder=4, alpha=0.85)
        self._artists.append(line)

        # Waypoint markers every ~10 % of path length
        step = max(1, len(path) // 10)
        wpm, = self.ax.plot(wxs[::step], wys[::step], 's',
                            color='#2980b9', markersize=7, zorder=5, alpha=0.9)
        self._artists.append(wpm)

        length = sum(
            np.hypot(wxs[i]-wxs[i-1], wys[i]-wys[i-1])
            for i in range(1, len(wxs))
        )
        self.ax.set_title(
            f'Path found — {length:.2f} m | Click to reset', fontsize=11
        )


if __name__ == '__main__':
    PathPlanner()
