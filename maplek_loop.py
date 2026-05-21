"""
maplek_loop.py  –  self-adaptive path planning loop

Subscribes to : path_maker:events   (init / obstacle messages)
Publishes to  : maplek_loop:paths   (new waypoints for each pair)

Start this before path_maker.py.
"""

import base64
import heapq
import json

import numpy as np
import redis

WORLD_SIZE  = 10.0
GRID_SIZE   = 200
BLOCK_SIZE  = 0.5
MIN_AB_DIST = 3.5
MIN_PT_DIST = 1.5

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
CH_EVENTS  = 'path_maker:events'   # subscribe here
CH_PATHS   = 'maplek_loop:paths'   # publish here

_MOVES = [
    (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
    ( 0, -1, 1.0),                  ( 0, 1, 1.0),
    ( 1, -1, 1.414), ( 1, 0, 1.0), ( 1, 1, 1.414),
]



def w2g(wx, wy):
    c = int(wx * GRID_SIZE / WORLD_SIZE)
    r = int((WORLD_SIZE - wy) * GRID_SIZE / WORLD_SIZE)
    return int(np.clip(r, 0, GRID_SIZE - 1)), int(np.clip(c, 0, GRID_SIZE - 1))


def g2w(r, c):
    return (c + 0.5) * WORLD_SIZE / GRID_SIZE, WORLD_SIZE - (r + 0.5) * WORLD_SIZE / GRID_SIZE


def astar(grid, start, goal):
    def h(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])
    open_set = [(h(start, goal), 0.0, start)]
    g_cost = {start: 0.0}
    prev = {}
    rows, cols = grid.shape
    while open_set:
        _, gc, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in prev:
                path.append(cur)
                cur = prev[cur]
            return [start] + path[::-1]
        if gc > g_cost.get(cur, float('inf')):
            continue
        for dr, dc, cost in _MOVES:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr, nc]:
                continue
            ng = gc + cost
            nb = (nr, nc)
            if ng < g_cost.get(nb, float('inf')):
                g_cost[nb] = ng
                prev[nb] = cur
                heapq.heappush(open_set, (ng + h(nb, goal), ng, nb))
    return None


class MaplekLoop:
    def __init__(self):
        self.base_grid    = None
        self.working_grid = None
        self.pairs         = []
        self.current_paths = []

        self._redis  = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(CH_EVENTS)

        print(f'[MaplekLoop] Subscribed to "{CH_EVENTS}", publishing on "{CH_PATHS}"')
        print('[MaplekLoop] Waiting for path_maker...')
        self._run()

    # ── main loop ─────────────────────────────────────────────────────────────

    def _run(self):
        try:
            for msg in self._pubsub.listen():
                if msg and msg['type'] == 'message':
                    try:
                        self._dispatch(json.loads(msg['data']))
                    except Exception as e:
                        print(f'[MaplekLoop] error: {e}')
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _shutdown(self):
        print('[MaplekLoop] Shutting down...')
        self._pubsub.unsubscribe()
        self._redis.close()
        print('[MaplekLoop] Bye.')

    def _dispatch(self, msg):
        t = msg.get('type')
        if t == 'init':
            self._on_init(msg)
        elif t == 'obstacle':
            self._on_obstacle(msg['x'], msg['y'])
        elif t == 'quit':
            raise KeyboardInterrupt

    def _publish(self, payload):
        self._redis.publish(CH_PATHS, json.dumps(payload))

    # ── event handlers ────────────────────────────────────────────────────────

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
        raise RuntimeError('Could not place B point — map too cluttered.')

    def _on_init(self, msg):
        shape = tuple(msg['grid_shape'])
        self.base_grid = np.frombuffer(
            base64.b64decode(msg['grid']), dtype=np.uint8
        ).reshape(shape).copy()
        self.working_grid = self.base_grid.copy()

        a_points = [(pt[0], pt[1]) for pt in msg['a_points']]
        self.pairs = []
        all_pts = list(a_points)
        for a_pt in a_points:
            b = self._random_free_point(all_pts, max(MIN_AB_DIST, MIN_PT_DIST))
            while np.hypot(a_pt[0] - b[0], a_pt[1] - b[1]) < MIN_AB_DIST:
                b = self._random_free_point(all_pts, MIN_PT_DIST)
            all_pts.append(b)
            self.pairs.append((a_pt, b))

        self.current_paths = []
        for i, (a_pt, b_pt) in enumerate(self.pairs):
            path = astar(self.working_grid, w2g(*a_pt), w2g(*b_pt))
            self.current_paths.append(path)
            waypoints = [list(g2w(r, c)) for r, c in path] if path else []
            self._publish({
                'type': 'new_path',
                'pair_id': i,
                'b_point': list(b_pt),
                'waypoints': waypoints,
            })
            status = f'{len(waypoints)} waypoints' if waypoints else 'blocked'
            print(f'[MaplekLoop] Pair {i+1} → B{i+1} placed, path: {status}')

    def _on_obstacle(self, wx, wy):
        half = BLOCK_SIZE / 2
        r0, c0 = w2g(wx - half, wy + half)
        r1, c1 = w2g(wx + half, wy - half)
        r0, r1 = min(r0, r1), max(r0, r1) + 1
        c0, c1 = min(c0, c1), max(c0, c1) + 1
        self.working_grid[r0:r1, c0:c1] = 1

        for i, path in enumerate(self.current_paths):
            if path is None or self._path_blocked(path):
                print(f'[MaplekLoop] Pair {i+1} path blocked — rerouting...')
                a_pt, b_pt = self.pairs[i]
                new_path = astar(self.working_grid, w2g(*a_pt), w2g(*b_pt))
                self.current_paths[i] = new_path
                waypoints = [list(g2w(r, c)) for r, c in new_path] if new_path else []
                self._publish({'type': 'new_path', 'pair_id': i, 'waypoints': waypoints})
                status = f'{len(waypoints)} waypoints' if waypoints else 'blocked'
                print(f'[MaplekLoop] Pair {i+1} new path: {status}')

    def _path_blocked(self, path):
        return any(self.working_grid[r, c] for r, c in path)


if __name__ == '__main__':
    MaplekLoop()
