"""
path_maker.py  –  visualisation + user interaction layer

Publishes to  : path_maker:events   (init / obstacle messages)
Subscribes to : maplek_loop:paths   (new waypoints from the planner)

Start order: launch maplek_loop.py first, then this file.
"""

import json
import queue
import threading

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import redis
from PIL import Image
from scipy.ndimage import binary_dilation

MAP_FILE     = 'robot_map.png'
WORLD_SIZE   = 10.0
GRID_SIZE    = 200
ROBOT_RADIUS = 0.15
BLOCK_SIZE   = 0.5
MIN_AB_DIST  = 3.5
MIN_PT_DIST  = 1.5

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
CH_EVENTS  = 'path_maker:events'   # publish here
CH_PATHS   = 'maplek_loop:paths'   # subscribe here

PAIR_STYLES = [
    ('#e74c3c', '#c0392b', '#e74c3c'),
    ('#3498db', '#1a5276', '#3498db'),
    ('#2ecc71', '#1e8449', '#2ecc71'),
]


def load_map():
    img = Image.open(MAP_FILE).convert('L')
    raw = np.array(img)
    small = np.array(img.resize((GRID_SIZE, GRID_SIZE), Image.LANCZOS))
    occupied = (small < 128).astype(np.uint8)
    rc = int(ROBOT_RADIUS * GRID_SIZE / WORLD_SIZE)
    if rc > 0:
        occupied = binary_dilation(
            occupied, structure=np.ones((2 * rc + 1, 2 * rc + 1))
        ).astype(np.uint8)
    return raw, occupied


def w2g(wx, wy):
    c = int(wx * GRID_SIZE / WORLD_SIZE)
    r = int((WORLD_SIZE - wy) * GRID_SIZE / WORLD_SIZE)
    return int(np.clip(r, 0, GRID_SIZE - 1)), int(np.clip(c, 0, GRID_SIZE - 1))


class PathMaker:
    def __init__(self):
        self.raw_img, self.base_grid = load_map()

        self._path_artists     = []
        self._block_artists    = []
        self._endpoint_artists = []
        self.pairs             = []
        self.current_paths     = [None, None, None]

        self._msg_queue = queue.Queue()

        self._redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(CH_PATHS)

        threading.Thread(target=self._recv_loop, daemon=True).start()

        self._setup_figure()
        self._randomize_endpoints()
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()

    # ── Redis I/O ─────────────────────────────────────────────────────────────

    def _recv_loop(self):
        for msg in self._pubsub.listen():
            if msg and msg['type'] == 'message':
                try:
                    self._msg_queue.put(json.loads(msg['data']))
                except Exception as e:
                    print(f'[PathMaker] recv error: {e}')

    def _publish(self, payload):
        self._redis.publish(CH_EVENTS, json.dumps(payload))

    # ── matplotlib setup ──────────────────────────────────────────────────────

    def _setup_figure(self):
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
                      'Click → place obstacle & reroute all   |   R → reset   |   Q → quit',
                      ha='center', fontsize=9, color='gray')

        self._timer = self.fig.canvas.new_timer(interval=100)
        self._timer.add_callback(self._drain_queue)
        self._timer.start()

    def _drain_queue(self):
        updated = False
        while not self._msg_queue.empty():
            msg = self._msg_queue.get_nowait()
            if msg.get('type') == 'new_path':
                self.current_paths[msg['pair_id']] = msg['waypoints']
                updated = True
        if updated:
            self._redraw_paths()

    # ── endpoints ─────────────────────────────────────────────────────────────

    def _random_free_point(self, avoid_pts, min_dist):
        rng = np.random.default_rng()
        for _ in range(20_000):
            wx = rng.uniform(0.5, WORLD_SIZE - 0.5)
            wy = rng.uniform(0.5, WORLD_SIZE - 0.5)
            if self.base_grid[w2g(wx, wy)]:
                continue
            if any(np.hypot(wx - ax, wy - ay) < min_dist for ax, ay in avoid_pts):
                continue
            return wx, wy
        raise RuntimeError('Could not place point — map too cluttered.')

    def _randomize_endpoints(self):
        self.pairs.clear()
        self.current_paths = [None, None, None]
        all_pts = []
        for _ in range(3):
            a = self._random_free_point(all_pts, MIN_PT_DIST)
            all_pts.append(a)
            b = self._random_free_point(all_pts + [a], max(MIN_AB_DIST, MIN_PT_DIST))
            while np.hypot(a[0] - b[0], a[1] - b[1]) < MIN_AB_DIST:
                b = self._random_free_point(all_pts + [a], MIN_PT_DIST)
            all_pts.append(b)
            self.pairs.append((a, b))

        self._clear_paths()
        self._draw_endpoints()
        self.fig.canvas.draw_idle()

        pairs_data = [[[a[0], a[1]], [b[0], b[1]]] for a, b in self.pairs]
        self._publish({'type': 'init', 'pairs': pairs_data})

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_endpoints(self):
        for art in self._endpoint_artists:
            art.remove()
        self._endpoint_artists.clear()
        for i, ((ax_w, ay_w), (bx_w, by_w)) in enumerate(self.pairs):
            ca, cb, _ = PAIR_STYLES[i]
            n = i + 1
            for wx, wy, color, label, marker in [
                (ax_w, ay_w, ca, f'A{n}', 'o'),
                (bx_w, by_w, cb, f'B{n}', 's'),
            ]:
                mk, = self.ax.plot(wx, wy, marker, color=color, markersize=13,
                                   zorder=8, markeredgecolor='white',
                                   markeredgewidth=1.8)
                tx = self.ax.text(wx + 0.15, wy + 0.15, label, color=color,
                                  fontsize=11, fontweight='bold', zorder=9)
                self._endpoint_artists.extend([mk, tx])

    def _clear_paths(self):
        for art in self._path_artists:
            art.remove()
        self._path_artists.clear()

    def _redraw_paths(self):
        self._clear_paths()
        lengths = []
        for i, waypoints in enumerate(self.current_paths):
            if waypoints is None:
                lengths.append(f'P{i+1}: ...')
            elif waypoints:
                _, _, path_color = PAIR_STYLES[i]
                wxs = [p[0] for p in waypoints]
                wys = [p[1] for p in waypoints]
                line, = self.ax.plot(wxs, wys, '-', color=path_color,
                                     linewidth=2.5, zorder=4, alpha=0.8)
                step = max(1, len(waypoints) // 12)
                dots, = self.ax.plot(wxs[::step], wys[::step], 's',
                                     color=path_color, markersize=6,
                                     zorder=5, alpha=0.85)
                self._path_artists.extend([line, dots])
                length = sum(np.hypot(wxs[j] - wxs[j-1], wys[j] - wys[j-1])
                             for j in range(1, len(wxs)))
                lengths.append(f'P{i+1}: {length:.1f}m')
            else:
                lengths.append(f'P{i+1}: blocked')
        self.ax.set_title('  |  '.join(lengths), fontsize=10)
        self.fig.canvas.draw_idle()

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        wx, wy = event.xdata, event.ydata
        half = BLOCK_SIZE / 2
        rect = mpatches.FancyBboxPatch(
            (wx - half, wy - half), BLOCK_SIZE, BLOCK_SIZE,
            boxstyle='square,pad=0', facecolor='#2c3e50',
            edgecolor='#7f8c8d', linewidth=1, zorder=6
        )
        self.ax.add_patch(rect)
        self._block_artists.append(rect)
        self.fig.canvas.draw_idle()
        self._publish({'type': 'obstacle', 'x': wx, 'y': wy})

    def _on_key(self, event):
        key = event.key.lower()
        if key == 'r':
            for art in self._block_artists:
                art.remove()
            self._block_artists.clear()
            self._randomize_endpoints()
        elif key == 'q':
            self._pubsub.unsubscribe()
            self._redis.close()
            plt.close(self.fig)


if __name__ == '__main__':
    PathMaker()
