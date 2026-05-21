"""
maplek_loop.py  –  self-adaptive path planning loop (MAPE-K architecture)

MAPE-K phases
─────────────
  Monitor    – reads incoming Redis topic messages and writes raw data to the
               Knowledge Base (in-memory + Redis kb:* keys)
  Analysis   – inspects the KB to decide which robots need a new path and why
  Plan       – runs A* for each flagged robot, returns candidate paths
  Legitimate – checks candidate paths for inter-robot timestep collisions;
               replans lower-priority robots until paths are conflict-free
               (or the retry limit is exhausted)
  Execute    – publishes validated paths to /robot_N/follow_waypoints and
               commits them back to the Knowledge Base

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

MAX_REPLAN_ATTEMPTS = 5   # maximum Legitimate retries per event

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

# ── Redis keys written by Monitor (Knowledge Base) ────────────────────────────
KB_MAP_INFO      = 'kb:map_info'
KB_ROBOT_POSES   = 'kb:robot_poses'
KB_GOAL_POSES    = 'kb:goal_poses'
KB_HUMAN_OBS     = 'kb:human_obstacle'
KB_PATH_LENGTHS  = 'kb:path_lengths'

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


# ── Knowledge Base ─────────────────────────────────────────────────────────────

class KnowledgeBase:
    """Shared state consumed and updated by every MAPE-K phase."""

    def __init__(self):
        self.base_grid    = None   # original grid from /map (no human obstacles)
        self.working_grid = None   # base_grid + accumulated human obstacle cells
        self.robot_poses  = {}     # robot_id → (wx, wy)
        self.goal_poses   = {}     # robot_id → (wx, wy)
        self.current_paths = {}    # robot_id → grid-coord path (committed after Execute)
        self.human_pos    = None   # (wx, wy, vx, vy) of latest human obstacle

        # readiness flags – all three must be True before initial paths are computed
        self.map_ready   = False
        self.poses_ready = False
        self.goals_ready = False


# ── Monitor ────────────────────────────────────────────────────────────────────

def monitor(channel, msg, kb: KnowledgeBase, redis_client) -> str | None:
    """
    MAPE-K Monitor phase.

    Parses the raw topic message, updates the in-memory Knowledge Base, and
    mirrors JSON-serialisable fields to Redis kb:* keys so external tools can
    inspect current state.

    Returns a short event tag: 'map' | 'poses' | 'goals' | 'human' | 'quit'
    or None if the channel is unrecognised.
    """
    if channel == T_MAP:
        shape = tuple(msg['data_shape'])
        kb.base_grid    = np.frombuffer(
            base64.b64decode(msg['data']), dtype=np.uint8
        ).reshape(shape).copy()
        kb.working_grid  = kb.base_grid.copy()
        kb.current_paths.clear()
        kb.map_ready   = True
        kb.poses_ready = False   # force re-receipt of poses after new map
        kb.goals_ready = False
        redis_client.set(KB_MAP_INFO, json.dumps(msg['info']))
        print(f'[Monitor] /map — '
              f'{msg["info"]["width"]}×{msg["info"]["height"]} '
              f'@ {msg["info"]["resolution"]:.3f} m/cell — grid reset')
        return 'map'

    elif channel == T_ROBOT_POSES:
        kb.robot_poses  = {p['robot_id']: (p['x'], p['y']) for p in msg['poses']}
        kb.poses_ready  = True
        redis_client.set(KB_ROBOT_POSES, json.dumps(
            {str(k): list(v) for k, v in kb.robot_poses.items()}
        ))
        print('[Monitor] /robot_poses — '
              + ', '.join(f'R{r}=({x:.2f},{y:.2f})' for r, (x, y) in kb.robot_poses.items()))
        return 'poses'

    elif channel == T_GOAL_POSES:
        kb.goal_poses   = {g['robot_id']: (g['x'], g['y']) for g in msg['goals']}
        kb.goals_ready  = True
        redis_client.set(KB_GOAL_POSES, json.dumps(
            {str(k): list(v) for k, v in kb.goal_poses.items()}
        ))
        print('[Monitor] /goal_poses — '
              + ', '.join(f'G{r}=({x:.2f},{y:.2f})' for r, (x, y) in kb.goal_poses.items()))
        return 'goals'

    elif channel == T_HUMAN_OBS:
        wx = msg['pose']['x']
        wy = msg['pose']['y']
        vx = msg['velocity']['vx']
        vy = msg['velocity']['vy']
        kb.human_pos = (wx, wy, vx, vy)
        redis_client.set(KB_HUMAN_OBS, json.dumps(msg))
        print(f'[Monitor] /human_obstacle at ({wx:.2f},{wy:.2f})  vel=({vx:.2f},{vy:.2f})')
        return 'human'

    elif channel == T_QUIT:
        return 'quit'

    return None


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze(kb: KnowledgeBase, event: str) -> set:
    """
    MAPE-K Analysis phase.

    Decides which robots need a new path.

    • map/poses/goals events: triggers initial planning once all three sources
      are available.
    • human event: marks the new obstacle cells in working_grid, then returns
      the subset of robots whose committed paths pass through those cells.

    Returns a set of robot_ids that must be replanned (may be empty).
    """
    if event in ('map', 'poses', 'goals'):
        if not (kb.map_ready and kb.poses_ready and kb.goals_ready):
            return set()
        robot_ids = set(kb.robot_poses.keys()) & set(kb.goal_poses.keys())
        print(f'[Analysis] All data ready — robots {sorted(robot_ids)} flagged for initial planning')
        return robot_ids

    elif event == 'human':
        wx, wy, _, _ = kb.human_pos
        half = BLOCK_SIZE / 2
        r0, c0 = w2g(wx - half, wy + half)
        r1, c1 = w2g(wx + half, wy - half)
        r0, r1 = min(r0, r1), max(r0, r1) + 1
        c0, c1 = min(c0, c1), max(c0, c1) + 1
        kb.working_grid[r0:r1, c0:c1] = 1   # commit obstacle to KB grid

        blocked = set()
        for robot_id, path in kb.current_paths.items():
            if path is not None and any(kb.working_grid[r, c] for r, c in path):
                blocked.add(robot_id)
                print(f'[Analysis] Robot {robot_id} path blocked by human — flagged for replan')
        return blocked

    return set()


# ── Plan ───────────────────────────────────────────────────────────────────────

def plan(kb: KnowledgeBase, robot_ids: set) -> dict:
    """
    MAPE-K Plan phase.

    Runs A* on the current working_grid for each robot in robot_ids.
    Returns a candidate_paths dict {robot_id: path} without modifying the KB
    — paths are only committed to the KB by Execute after Legitimate approves.
    """
    candidate_paths = {}
    for robot_id in sorted(robot_ids):
        if robot_id not in kb.robot_poses or robot_id not in kb.goal_poses:
            continue
        path = astar(
            kb.working_grid,
            w2g(*kb.robot_poses[robot_id]),
            w2g(*kb.goal_poses[robot_id]),
        )
        candidate_paths[robot_id] = path
        status = f'{len(path)} waypoints' if path else 'no path'
        print(f'[Plan] Robot {robot_id} → {status}')
    return candidate_paths


# ── Legitimate ─────────────────────────────────────────────────────────────────

def _check_timestep_collisions(paths: dict) -> set:
    """
    Scan all robot-pairs for cell-occupancy collisions at the same timestep,
    including head-on swap collisions.

    Robot with the lower ID has higher priority and keeps its path; the robot
    with the higher ID is added to the returned needs_replan set.
    """
    robot_ids  = sorted(r for r, p in paths.items() if p is not None)
    needs_replan = set()

    for i in range(len(robot_ids)):
        for j in range(i + 1, len(robot_ids)):
            id_a, id_b = robot_ids[i], robot_ids[j]   # id_a has higher priority
            path_a, path_b = paths[id_a], paths[id_b]
            max_t = max(len(path_a), len(path_b))

            for t in range(max_t):
                pos_a = path_a[min(t, len(path_a) - 1)]
                pos_b = path_b[min(t, len(path_b) - 1)]

                if pos_a == pos_b:
                    print(f'[Legitimate] Collision R{id_a}↔R{id_b} at t={t} cell={pos_a}')
                    needs_replan.add(id_b)
                    break

                # swap collision: A moves into B's previous cell and vice-versa
                if t > 0:
                    prev_a = path_a[min(t - 1, len(path_a) - 1)]
                    prev_b = path_b[min(t - 1, len(path_b) - 1)]
                    if pos_a == prev_b and pos_b == prev_a:
                        print(f'[Legitimate] Swap collision R{id_a}↔R{id_b} at t={t}')
                        needs_replan.add(id_b)
                        break

    return needs_replan


def legitimate(kb: KnowledgeBase, candidate_paths: dict) -> dict:
    """
    MAPE-K Legitimate phase.

    Iterates up to MAX_REPLAN_ATTEMPTS times:
      1. Check all path pairs for timestep collisions.
      2. If clean, return paths.
      3. For each lower-priority robot flagged for replan, build a temporary
         grid that additionally blocks all cells occupied by higher-priority
         robots, then re-run A* for the flagged robot.

    Returns the validated paths dict.  Paths that could not be de-conflicted
    within the attempt limit are kept as-is with a warning logged.
    """
    paths = dict(candidate_paths)
    needs_replan: set = set()

    for attempt in range(MAX_REPLAN_ATTEMPTS):
        needs_replan = _check_timestep_collisions(paths)
        if not needs_replan:
            print(f'[Legitimate] All paths conflict-free'
                  + (f' after {attempt} replan(s)' if attempt else ''))
            break

        print(f'[Legitimate] Attempt {attempt + 1}/{MAX_REPLAN_ATTEMPTS} — '
              f'replanning robots {sorted(needs_replan)}')

        robot_ids_sorted = sorted(paths.keys())
        for robot_id in sorted(needs_replan):
            if robot_id not in kb.robot_poses or robot_id not in kb.goal_poses:
                continue

            # block every cell used by any higher-priority robot's path
            temp_grid = kb.working_grid.copy()
            for other_id in robot_ids_sorted:
                if other_id >= robot_id:
                    continue
                other_path = paths.get(other_id)
                if other_path:
                    for r, c in other_path:
                        temp_grid[r, c] = 1

            new_path = astar(
                temp_grid,
                w2g(*kb.robot_poses[robot_id]),
                w2g(*kb.goal_poses[robot_id]),
            )
            paths[robot_id] = new_path
            status = f'{len(new_path)} waypoints' if new_path else 'blocked'
            print(f'[Legitimate] Robot {robot_id} replanned → {status}')
    else:
        remaining = _check_timestep_collisions(paths)
        if remaining:
            print(f'[Legitimate] WARNING: unresolved conflicts after '
                  f'{MAX_REPLAN_ATTEMPTS} attempts for robots {sorted(remaining)}')

    return paths


# ── Execute ────────────────────────────────────────────────────────────────────

def _publish_waypoints(redis_client, robot_id: int, path):
    poses = [{'x': x, 'y': y} for x, y in
             (g2w(r, c) for r, c in path)] if path else []
    redis_client.publish(T_WAYPOINTS[robot_id - 1], json.dumps({
        'header':   {'stamp': _now(), 'frame_id': 'map'},
        'robot_id': robot_id,
        'poses':    poses,
    }))


def execute(redis_client, validated_paths: dict, kb: KnowledgeBase):
    """
    MAPE-K Execute phase.

    Commits validated paths to the Knowledge Base and publishes each path to
    the corresponding /robot_N/follow_waypoints topic.  Also writes a path-
    length summary to kb:path_lengths for external observability.
    """
    for robot_id, path in validated_paths.items():
        kb.current_paths[robot_id] = path
        _publish_waypoints(redis_client, robot_id, path)
        status = f'{len(path)} waypoints' if path else 'blocked'
        print(f'[Execute] Robot {robot_id} → {status}')

    redis_client.set(KB_PATH_LENGTHS, json.dumps({
        str(rid): len(p) if p else 0
        for rid, p in kb.current_paths.items()
    }))


# ── Orchestrator ───────────────────────────────────────────────────────────────

class MaplekLoop:
    """
    Thin orchestrator that owns the Redis connection and pub/sub loop.
    All logic is delegated to the five MAPE-K functions above.
    """

    def __init__(self):
        self._kb     = KnowledgeBase()
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

    # ── startup ───────────────────────────────────────────────────────────────

    def _load_cached_state(self):
        """Read state keys written by path_maker so we can start in any order."""
        raw_map   = self._redis.get(K_MAP)
        raw_poses = self._redis.get(K_ROBOT_POSES)
        raw_goals = self._redis.get(K_GOAL_POSES)

        if raw_map and raw_poses and raw_goals:
            print('[MaplekLoop] Cached state found — loading without waiting for path_maker.')
            for channel, raw in (
                (T_MAP,         raw_map),
                (T_ROBOT_POSES, raw_poses),
                (T_GOAL_POSES,  raw_goals),
            ):
                self._mape_k(channel, json.loads(raw))
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
                        self._mape_k(msg['channel'], json.loads(msg['data']))
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

    # ── MAPE-K pipeline ───────────────────────────────────────────────────────

    def _mape_k(self, channel: str, msg: dict):
        # Monitor
        event = monitor(channel, msg, self._kb, self._redis)
        if event == 'quit':
            raise KeyboardInterrupt
        if event is None:
            return

        # Analysis
        robot_ids = analyze(self._kb, event)
        if not robot_ids:
            return

        # Plan
        candidate_paths = plan(self._kb, robot_ids)
        if not candidate_paths:
            return

        # Legitimate
        validated_paths = legitimate(self._kb, candidate_paths)

        # Execute
        execute(self._redis, validated_paths, self._kb)


if __name__ == '__main__':
    MaplekLoop()
