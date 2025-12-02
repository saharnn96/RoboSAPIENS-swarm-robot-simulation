from irsim.lib import register_behavior
from irsim.util.util import relative_position, WrapToPi
from irsim.lib.behavior.behavior_methods import DiffRVO
import numpy as np

@register_behavior("diff", "rvo_custom")
def beh_diff_rvo(ego_object, external_objects, **kwargs):

    rvo_neighbor = [obj.rvo_neighbor_state for obj in external_objects]
    rvo_state = ego_object.rvo_state
    fov_objects = []
    for obj in external_objects:
        relative_pos = relative_position(ego_object.state, obj.state)
        distance, angle = relative_pos
        if distance <= ego_object.fov_radius and abs(WrapToPi(angle - ego_object.state[2, 0])) <= ego_object.fov / 2:
            fov_objects.append(obj)
    vxmax = kwargs.get("vxmax", 1.5)
    vymax = kwargs.get("vymax", 1.5)
    acceler = kwargs.get("acceler", 1.0)
    factor = kwargs.get("factor", 1.0)
    mode = kwargs.get("mode", "rvo")
    for obj in fov_objects:
        if obj.color == "pink":  # Assuming the object has a 'color' attribute
            vxmax *= 0.5  # Reduce speed limits by half
            vymax *= 0.5
            print("Human detected, reducing speed limits.")
            break
    behavior_vel = DiffRVO(rvo_state, rvo_neighbor, vxmax, vymax, acceler, factor, mode)
    return behavior_vel

def omni_to_diff(
    state_ori, vel_omni, w_max=1.5, guarantee_time=0.2, tolerance=0.1, mini_speed=0.02
):
    """
    Convert omnidirectional velocity to differential velocity.

    Args:
        state_ori (float): Orientation angle.
        vel_omni (np.array): Omnidirectional velocity [vx, vy] (2x1).
        w_max (float): Maximum angular velocity.
        guarantee_time (float): Time to guarantee velocity.
        tolerance (float): Angular tolerance.
        mini_speed (float): Minimum speed threshold.

    Returns:
        np.array: Differential velocity [linear, angular] (2x1).
    """
    if isinstance(vel_omni, list):
        vel_omni = np.array(vel_omni).reshape((2, 1))

    speed = np.sqrt(vel_omni[0, 0] ** 2 + vel_omni[1, 0] ** 2)

    if speed <= mini_speed:
        return np.zeros((2, 1))

    vel_radians = np.atan2(vel_omni[1, 0], vel_omni[0, 0])
    robot_radians = state_ori
    diff_radians = robot_radians - vel_radians

    diff_radians = WrapToPi(diff_radians)

    if abs(diff_radians) < tolerance:
        w = 0
    else:
        w = -diff_radians / guarantee_time
        if w > w_max:
            w = w_max
        if w < -w_max:
            w = -w_max

    v = speed * np.cos(diff_radians)
    if v < 0:
        v = 0

    return np.array([[v], [w]])

@register_behavior("diff", "dash_custom")
def beh_diff_dash(ego_object, external_objects=[], **kwargs):

    print("This is a custom behavior example for differential drive with dash2")

    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    _, max_vel = ego_object.get_vel_range()
    # angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    
    behavior_vel = DiffDash2(state, goal, max_vel, goal_threshold=goal_threshold)

    return behavior_vel


def DiffDash2(state, goal, max_vel, goal_threshold=0.1, angle_tolerance=0.2):
    """
    Calculate the differential drive velocity to reach a goal.

    Args:
        state (np.array): Current state [x, y, theta] (3x1).
        goal (np.array): Goal position [x, y, theta] (3x1).
        max_vel (np.array): Maximum velocity [linear, angular] (2x1).
        goal_threshold (float): Distance threshold to consider goal reached (default 0.1).
        angle_tolerance (float): Allowable angular deviation (default 0.2).

    Returns:
        np.array: Velocity [linear, angular] (2x1).
    """
    distance, radian = relative_position(state, goal)

    if distance < goal_threshold:
        return np.zeros((2, 1))

    diff_radian = WrapToPi(radian - state[2, 0])
    linear = max_vel[0, 0] * np.cos(diff_radian)

    if abs(diff_radian) < angle_tolerance:
        angular = 0
    else:
        angular = max_vel[1, 0] * np.sign(diff_radian)

    return np.array([[linear], [angular]])