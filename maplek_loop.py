"""
maplek_loop.py  –  self-adaptive path planning loop

Subscribes (ROS2-style topics via Redis):
  /map                       nav_msgs/OccupancyGrid  – inflated obstacle grid
  /robot_poses               geometry_msgs/PoseArray – robot start positions
  /goal_poses                geometry_msgs/PoseArray – robot goal positions
  /human_obstacle            PoseWithVelocity        – moving obstacle with velocity
  /system/quit               –                       – shutdown signal

Publishes (ROS2-style topics via Redis):
  /robot_N/follow_waypoints  nav_msgs/Path           – planned path per robot (N=1,2,3)

Paths are (re)computed whenever all three of /map, /robot_poses, /goal_poses have
been received.  Each /human_obstacle message triggers a grid update and rereroutes
only the robots whose current paths pass through the newly blocked cells.
"""

import base64
import heapq
import json
import time

import numpy as np
import redis

WORLD_SIZE = 10.0
GRID_SIZE  = 200
BLOCK_SIZE = 0.5
NUM_ROBOTS = 3

REDIS_HOST = 'localhost'
REDIS_PORT = 6379

# ── topic names (mirrors ROS2 topic convention) ───────────────────────────────
T_MAP         = '/map'
T_ROBOT_POSES = '/robot_poses'
T_GOAL_POSES  = '/goal_poses'
T_HUMAN_OBS   = '/human_obstacle'
T_QUIT        = '/system/quit'
T_WAYPOINTS   = [f'/robot_{i+1}/follow_waypoints' for i in range(NUM_ROBOTS)]

# ── Redis keys written by path_maker (read on late startup) ───────────────────
K_MAP         = 'state:map'
K_ROBOT_POSES = 'state:robot_poses'
K_GOAL_POSES  = 'state:goal_poses'

_MOVES = [
    (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
    ( 0, -1, 1.0),                  ( 0, 1, 1.0),
    ( 1, -1, 1.414), ( 1, 0, 1.0), ( 1, 1, 1.414),
]


def _now():
    return time.time()


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
        self.robot_poses  = {}   # robot_id → (x, y)
        self.goal_poses   = {}   # robot_id → (x, y)
        self.current_paths = {}  # robot_id → grid-coord path (or None)

        # readiness flags — all three must be True before paths are computed
        self._map_ready   = False
        self._poses_ready = False
        self._goals_ready = False

        self._redis  = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(T_MAP, T_ROBOT_POSES, T_GOAL_POSES, T_HUMAN_OBS, T_QUIT)

        print('[MaplekLoop] Subscribed to:')
        for t in (T_MAP, T_ROBOT_POSES, T_GOAL_POSES, T_HUMAN_OBS, T_QUIT):
            print(f'  SUB  {t}')
        for t in T_WAYPOINTS:
            print(f'  PUB  {t}')

        self._load_cached_state()
        self._run()

    def _load_cached_state(self):
        """Read state keys written by path_maker so we can start in any order."""
        raw_map   = self._redis.get(K_MAP)
        raw_poses = self._redis.get(K_ROBOT_POSES)
        raw_goals = self._redis.get(K_GOAL_POSES)

        if raw_map and raw_poses and raw_goals:
            print('[MaplekLoop] Cached state found — loading without waiting for path_maker.')
            self._on_map(json.loads(raw_map))
            self._on_robot_poses(json.loads(raw_poses))
            self._on_goal_poses(json.loads(raw_goals))
        else:
            missing = [k for k, v in
                       ((K_MAP, raw_map), (K_ROBOT_POSES, raw_poses), (K_GOAL_POSES, raw_goals))
                       if not v]
            print(f'[MaplekLoop] No cached state ({", ".join(missing)} missing). '
                  'Waiting for path_maker...')

    # ── main loop ─────────────────────────────────────────────────────────────

    def _run(self):
        try:
            for msg in self._pubsub.listen():
                if msg and msg['type'] == 'message':
                    try:
                        self._dispatch(msg['channel'], json.loads(msg['data']))
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

    def _dispatch(self, channel, msg):
        if   channel == T_MAP:         self._on_map(msg)
        elif channel == T_ROBOT_POSES: self._on_robot_poses(msg)
        elif channel == T_GOAL_POSES:  self._on_goal_poses(msg)
        elif channel == T_HUMAN_OBS:   self._on_human_obstacle(msg)
        elif channel == T_QUIT:        raise KeyboardInterrupt

    # ── topic publisher ───────────────────────────────────────────────────────

    def _publish_waypoints(self, robot_id, path):
        poses = [{'x': x, 'y': y} for x, y in
                 (g2w(r, c) for r, c in path)] if path else []
        self._redis.publish(T_WAYPOINTS[robot_id - 1], json.dumps({
            'header':   {'stamp': _now(), 'frame_id': 'map'},
            'robot_id': robot_id,
            'poses':    poses,
        }))

    # ── topic handlers ────────────────────────────────────────────────────────

    def _on_map(self, msg):
        shape = tuple(msg['data_shape'])
        self.base_grid    = np.frombuffer(
            base64.b64decode(msg['data']), dtype=np.uint8
        ).reshape(shape).copy()
        self.working_grid = self.base_grid.copy()
        self.current_paths.clear()
        # reset pose flags so paths are recomputed once new poses also arrive
        self._map_ready   = True
        self._poses_ready = False
        self._goals_ready = False
        print(f'[MaplekLoop] /map received  '
              f'({msg["info"]["width"]}×{msg["info"]["height"]} '
              f'@ {msg["info"]["resolution"]:.3f} m/cell) — grid reset.')
        self._try_compute_paths()

    def _on_robot_poses(self, msg):
        self.robot_poses  = {p['robot_id']: (p['x'], p['y']) for p in msg['poses']}
        self._poses_ready = True
        print(f'[MaplekLoop] /robot_poses received — '
              + ', '.join(f'R{r}=({x:.2f},{y:.2f})' for r,(x,y) in self.robot_poses.items()))
        self._try_compute_paths()

    def _on_goal_poses(self, msg):
        self.goal_poses   = {g['robot_id']: (g['x'], g['y']) for g in msg['goals']}
        self._goals_ready = True
        print(f'[MaplekLoop] /goal_poses received  — '
              + ', '.join(f'G{r}=({x:.2f},{y:.2f})' for r,(x,y) in self.goal_poses.items()))
        self._try_compute_paths()

    def _try_compute_paths(self):
        if not (self._map_ready and self._poses_ready and self._goals_ready):
            return
        print('[MaplekLoop] All data ready — computing initial paths...')
        self.current_paths.clear()
        for robot_id in sorted(self.robot_poses):
            if robot_id not in self.goal_poses:
                continue
            path = astar(self.working_grid,
                         w2g(*self.robot_poses[robot_id]),
                         w2g(*self.goal_poses[robot_id]))
            self.current_paths[robot_id] = path
            self._publish_waypoints(robot_id, path)
            status = f'{len(path)} waypoints' if path else 'blocked'
            print(f'[MaplekLoop] Robot {robot_id} → {status}')

    def _on_human_obstacle(self, msg):
        wx = msg['pose']['x']
        wy = msg['pose']['y']
        vx = msg['velocity']['vx']
        vy = msg['velocity']['vy']
        print(f'[MaplekLoop] /human_obstacle at ({wx:.2f}, {wy:.2f})  '
              f'vel=({vx:.2f}, {vy:.2f})')

        # mark obstacle cells in working grid
        half = BLOCK_SIZE / 2
        r0, c0 = w2g(wx - half, wy + half)
        r1, c1 = w2g(wx + half, wy - half)
        r0, r1 = min(r0, r1), max(r0, r1) + 1
        c0, c1 = min(c0, c1), max(c0, c1) + 1
        self.working_grid[r0:r1, c0:c1] = 1

        for robot_id, path in list(self.current_paths.items()):
            if path is None or self._path_blocked(path):
                new_path = astar(self.working_grid,
                                 w2g(*self.robot_poses[robot_id]),
                                 w2g(*self.goal_poses[robot_id]))
                self.current_paths[robot_id] = new_path
                self._publish_waypoints(robot_id, new_path)
                status = f'{len(new_path)} waypoints' if new_path else 'blocked'
                print(f'[MaplekLoop] Robot {robot_id} rerouted → {status}')

    def _path_blocked(self, path):
        return any(self.working_grid[r, c] for r, c in path)


if __name__ == '__main__':
    MaplekLoop()
