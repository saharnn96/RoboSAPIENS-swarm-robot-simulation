"""
path_maker.py  –  visualisation + sensor simulation layer

Publishes  (ROS2-style topics via Redis):
  /map                       nav_msgs/OccupancyGrid  – inflated obstacle grid
  /robot_poses               geometry_msgs/PoseArray – robot start positions (A points)
  /goal_poses                geometry_msgs/PoseArray – robot goal positions  (B points)
  /human_obstacle            PoseWithVelocity        – clicked obstacle with velocity
  /system/quit               –                       – shutdown signal

Subscribes (ROS2-style topics via Redis):
  /robot_N/follow_waypoints  nav_msgs/Path           – planned path per robot (N=1,2,3)
"""

import base64
import json
import queue
import threading
import time

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
NUM_ROBOTS   = 3

REDIS_HOST = 'localhost'
REDIS_PORT = 6379

# ── topic names (mirrors ROS2 topic convention) ───────────────────────────────
T_MAP         = '/map'
T_ROBOT_POSES = '/robot_poses'
T_GOAL_POSES  = '/goal_poses'
T_HUMAN_OBS   = '/human_obstacle'
T_QUIT        = '/system/quit'
T_WAYPOINTS   = [f'/robot_{i+1}/follow_waypoints' for i in range(NUM_ROBOTS)]

# ── Redis keys for persisting latest state (allows maplek_loop to start late) ─
K_MAP         = 'state:map'
K_ROBOT_POSES = 'state:robot_poses'
K_GOAL_POSES  = 'state:goal_poses'

PAIR_STYLES = [
    ('#e74c3c', '#c0392b', '#e74c3c'),
    ('#3498db', '#1a5276', '#3498db'),
    ('#2ecc71', '#1e8449', '#2ecc71'),
]


def _now():
    return time.time()


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
        self.a_points      = []
        self.b_points      = []
        self.current_paths = [None] * NUM_ROBOTS

        self._msg_queue = queue.Queue()

        self._redis  = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(*T_WAYPOINTS)

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
                    data = json.loads(msg['data'])
                    data['_channel'] = msg['channel']
                    self._msg_queue.put(data)
                except Exception as e:
                    print(f'[PathMaker] recv error: {e}')

    def _publish(self, topic, payload):
        self._redis.publish(topic, json.dumps(payload))

    def _header(self):
        return {'stamp': _now(), 'frame_id': 'map'}

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
                      'Click → place human obstacle   |   R → reset   |   Q → quit',
                      ha='center', fontsize=9, color='gray')

        self._timer = self.fig.canvas.new_timer(interval=100)
        self._timer.add_callback(self._drain_queue)
        self._timer.start()

    def _drain_queue(self):
        updated = False
        while not self._msg_queue.empty():
            msg = self._msg_queue.get_nowait()
            ch = msg.get('_channel', '')
            for i, topic in enumerate(T_WAYPOINTS):
                if ch == topic:
                    poses = msg.get('poses', [])
                    self.current_paths[i] = [(p['x'], p['y']) for p in poses] if poses else []
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
            if any(np.hypot(wx - px, wy - py) < min_dist for px, py in avoid_pts):
                continue
            return wx, wy
        raise RuntimeError('Could not place point — map too cluttered.')

    def _randomize_endpoints(self):
        self.a_points = []
        self.b_points = []
        self.current_paths = [None] * NUM_ROBOTS
        all_pts = []
        for _ in range(NUM_ROBOTS):
            a = self._random_free_point(all_pts, MIN_PT_DIST)
            all_pts.append(a)
            b = self._random_free_point(all_pts + [a], max(MIN_AB_DIST, MIN_PT_DIST))
            while np.hypot(a[0] - b[0], a[1] - b[1]) < MIN_AB_DIST:
                b = self._random_free_point(all_pts + [a], MIN_PT_DIST)
            all_pts.append(b)
            self.a_points.append(a)
            self.b_points.append(b)

        self._clear_paths()
        self._draw_endpoints()
        self.fig.canvas.draw_idle()

        # publish in order: map first so maplek_loop resets, then poses
        self._publish_map()
        self._publish_robot_poses()
        self._publish_goal_poses()

    # ── topic publishers ──────────────────────────────────────────────────────

    def _publish_map(self):
        grid_b64 = base64.b64encode(self.base_grid.tobytes()).decode('ascii')
        payload = {
            'header': self._header(),
            'info': {
                'resolution': WORLD_SIZE / GRID_SIZE,   # metres per cell
                'width':      GRID_SIZE,
                'height':     GRID_SIZE,
            },
            'data':       grid_b64,
            'data_shape': list(self.base_grid.shape),
        }
        self._publish(T_MAP, payload)
        self._redis.set(K_MAP, json.dumps(payload))

    def _publish_robot_poses(self):
        payload = {
            'header': self._header(),
            'poses': [
                {'robot_id': i + 1, 'x': a[0], 'y': a[1]}
                for i, a in enumerate(self.a_points)
            ],
        }
        self._publish(T_ROBOT_POSES, payload)
        self._redis.set(K_ROBOT_POSES, json.dumps(payload))

    def _publish_goal_poses(self):
        payload = {
            'header': self._header(),
            'goals': [
                {'robot_id': i + 1, 'x': b[0], 'y': b[1]}
                for i, b in enumerate(self.b_points)
            ],
        }
        self._publish(T_GOAL_POSES, payload)
        self._redis.set(K_GOAL_POSES, json.dumps(payload))

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_endpoints(self):
        for art in self._endpoint_artists:
            art.remove()
        self._endpoint_artists.clear()
        for i, (ax_w, ay_w) in enumerate(self.a_points):
            ca, _, _ = PAIR_STYLES[i]
            mk, = self.ax.plot(ax_w, ay_w, 'o', color=ca, markersize=13,
                               zorder=8, markeredgecolor='white', markeredgewidth=1.8)
            tx = self.ax.text(ax_w + 0.15, ay_w + 0.15, f'A{i+1}', color=ca,
                              fontsize=11, fontweight='bold', zorder=9)
            self._endpoint_artists.extend([mk, tx])
        for i, (bx_w, by_w) in enumerate(self.b_points):
            _, cb, _ = PAIR_STYLES[i]
            mk, = self.ax.plot(bx_w, by_w, 's', color=cb, markersize=13,
                               zorder=8, markeredgecolor='white', markeredgewidth=1.8)
            tx = self.ax.text(bx_w + 0.15, by_w + 0.15, f'B{i+1}', color=cb,
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
                lengths.append(f'R{i+1}: ...')
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
                lengths.append(f'R{i+1}: {length:.1f}m')
            else:
                lengths.append(f'R{i+1}: blocked')
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

        # publish human obstacle — velocity is zero for a static click;
        # a real people-tracker node would fill vx/vy with measured values
        self._publish(T_HUMAN_OBS, {
            'header':   self._header(),
            'pose':     {'x': wx,  'y': wy},
            'velocity': {'vx': 0.0, 'vy': 0.0},
        })

    def _on_key(self, event):
        key = event.key.lower()
        if key == 'r':
            for art in self._block_artists:
                art.remove()
            self._block_artists.clear()
            self._randomize_endpoints()
        elif key == 'q':
            self._publish(T_QUIT, {'header': self._header(), 'reason': 'user_quit'})
            self._pubsub.unsubscribe()
            self._redis.close()
            plt.close(self.fig)


if __name__ == '__main__':
    PathMaker()
