#!/usr/bin/env python3
"""Measure chess pieces with the Hand-E (the robot moves - stay at the E-stop).

Above each piece the fingers step down 5 mm at a time and close/open at
every level: contact = the piece reaches that high, the stop position = its
width there.  The scan stops BODY_MARGIN below the first contact so the
gripper body (46.5 mm above the fingertips) never touches the piece's top.
Uses the running move_group (Cartesian paths, 5-10 % speed) and the
gripper_service of the normal bringup; no robot_control commands.

Board: clear it, put one piece of each type on the squares in PIECES.
The board corners (board_h8_xy / board_a1_xy) are read from the running
robot_control_node (fallback: the values below).

    python3 measure_pieces.py check              # no motion: services, targets
    python3 measure_pieces.py hover c3           # 15 cm above a square
    python3 measure_pieces.py measure [squares]  # -> ~/chess_piece_measure.json
    python3 measure_pieces.py lift [squares]     # grip at the chosen height, lift 3 cm
    python3 measure_pieces.py yaml [out.yaml]    # no robot: pieces.yaml from the scan

`yaml` picks for every piece the lowest scanned fingertip height that keeps
the gripper body GRIP_CLEARANCE above the piece's top (and >= 15 mm), and
the width measured there.  Lift-test the result (`lift` uses the heights in
LIFT_GRIPS - edit them) before using it for games.
"""
import json
import os
import sys
import time

import rclpy
from geometry_msgs.msg import Pose
from gripper_srv.srv import GripperService
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.node import Node
import tf2_ros

QUAT = (-0.0028119066264480352, 0.9999957084655762, -0.0007648332393728197, -0.00023792324645910412)
EE = "ur5e_robotiq_hande_end"
FRAME = "world"
H8 = [1.1099, 0.4478]
A1 = [0.7191, 0.0458]
HANDE_BODY = 0.0465          # gripper body above the fingertips [m]
GRIP_CLEARANCE = 0.010       # body above the piece's top when gripping [m]
SAFE_Z = 0.20
START_Z = 0.075
STEP = 0.005
BODY_MARGIN = 0.031          # scan at most this far below the first contact (gripper body!)
MIN_Z = 0.008
PIECES = {"c3": "K", "e3": "Q", "g3": "R", "c5": "B", "e5": "N", "g5": "P"}
OUT = os.path.expanduser("~/chess_piece_measure.json")
# grip heights tried by `lift` (the cell's wooden set, 2026-10-30)
LIFT_GRIPS = {"c3": 0.030, "e3": 0.020, "g3": 0.015, "c5": 0.015, "e5": 0.012, "g5": 0.015}


def square_xy(name):
    f, k = "abcdefgh".index(name[0]), int(name[1])
    r, c = 8 - k, f                                  # game frame of robot_control
    dx, dy = (H8[0] - A1[0]) / 7.0, (H8[1] - A1[1]) / 7.0
    urx, ury = 0.5 * (dy - dx), -0.5 * (dx + dy)
    ucx, ucy = -ury, urx
    return A1[0] + (r - 7) * urx + c * ucx, A1[1] + (r - 7) * ury + c * ucy


class M(Node):
    def __init__(self):
        super().__init__("chess_piece_measure")
        self.cart = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.exe = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self.grip = self.create_client(GripperService, "gripper_service")
        self.tfb = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfb, self)

    def wait(self, fut, t=30.0):
        end = time.time() + t
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not fut.done():
            raise RuntimeError("timeout")
        return fut.result()

    def pose(self):
        end = time.time() + 5
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                t = self.tfb.lookup_transform(FRAME, EE, rclpy.time.Time())
                return t.transform.translation
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError("no TF")

    def move(self, pts, vel=0.10):
        req = GetCartesianPath.Request()
        req.header.frame_id = FRAME
        req.group_name = "ur5e_arm"
        req.link_name = EE
        req.start_state.is_diff = True
        for x, y, z in pts:
            p = Pose()
            p.position.x, p.position.y, p.position.z = x, y, z
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = QUAT
            req.waypoints.append(p)
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = vel
        req.max_acceleration_scaling_factor = vel
        res = self.wait(self.cart.call_async(req))
        if res.fraction < 0.999:
            raise RuntimeError(f"cartesian path only {res.fraction:.2f} feasible")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        gh = self.wait(self.exe.send_goal_async(goal))
        if not gh.accepted:
            raise RuntimeError("trajectory rejected")
        r = self.wait(gh.get_result_async(), 60)
        if r.result.error_code.val != 1:
            raise RuntimeError(f"execution failed {r.result.error_code.val}")

    def gripper(self, pos, force=10):
        req = GripperService.Request()
        req.position, req.speed, req.force = pos, 30, force
        r = self.wait(self.grip.call_async(req), 10)
        return r.final_position, r.object_status


def pieces_yaml(results, margin=0.025):
    """pieces.yaml text from a measurement (see the module doc)."""
    lines = ["# measured with tools/measure_pieces.py - lift-test before use", "pieces:"]
    for sym in "KQBNRP":
        r = results.get(sym)
        if not r or r["first_contact_z"] is None:
            lines.append(f"  # {sym}: not measured")
            continue
        top = r["first_contact_z"] + STEP
        low = max(top - HANDE_BODY + GRIP_CLEARANCE, 0.015)
        cand = [lv for lv in r["levels"] if lv[3] is not None and lv[0] >= low - 1e-6]
        z, _, _, width = min(cand, key=lambda lv: lv[0])
        lines.append(f"  {sym}: {{height: {top:.3f}, grip_height: {z:.3f}, grip_width: {width / 1000:.4f}}}")
    lines.append(f"travel_margin: {margin}")
    return "\n".join(lines) + "\n"


def board_from_robot_control(node):
    """board_h8_xy / board_a1_xy of the running robot_control_node."""
    from rcl_interfaces.srv import GetParameters
    cli = node.create_client(GetParameters, "/robot_control_node/get_parameters")
    if not cli.wait_for_service(timeout_sec=3):
        print("robot_control_node not found - using the default board corners")
        return
    res = node.wait(cli.call_async(GetParameters.Request(names=["board_h8_xy", "board_a1_xy"])))
    h8, a1 = (list(v.double_array_value) for v in res.values)
    if len(h8) == 2 and len(a1) == 2:
        H8[:], A1[:] = h8, a1
        print(f"board from robot_control: H8 {H8}, A1 {A1}")


def main():
    if sys.argv[1] == "yaml":
        text = pieces_yaml(json.load(open(OUT)))
        if len(sys.argv) > 2:
            open(sys.argv[2], "w").write(text)
        print(text)
        return
    rclpy.init()
    n = M()
    board_from_robot_control(n)
    for c, name in ((n.cart, "cartesian"), (n.grip, "gripper")):
        if not c.wait_for_service(timeout_sec=5):
            sys.exit(f"no {name} service")
    if not n.exe.wait_for_server(timeout_sec=5):
        sys.exit("no execute_trajectory action")
    p = n.pose()
    print(f"TCP now {p.x:.4f} {p.y:.4f} {p.z:.4f}", flush=True)
    mode = sys.argv[1]
    if mode == "check":
        for sq in PIECES:
            print(sq, ["%.4f" % v for v in square_xy(sq)])
        return
    if mode == "lift":
        for sq in sys.argv[2:] or list(LIFT_GRIPS):
            x, y = square_xy(sq)
            z = LIFT_GRIPS[sq]
            p = n.pose()
            n.move([(p.x, p.y, max(p.z, SAFE_Z)), (x, y, SAFE_Z), (x, y, START_Z)])
            n.gripper(0, 5)
            n.move([(x, y, z)], vel=0.05)
            pos, st = n.gripper(255, 20)
            held = st == 2
            if held:
                n.move([(x, y, z + 0.03)], vel=0.05)
                time.sleep(1.0)
                n.move([(x, y, z + 0.002)], vel=0.05)
            n.gripper(0, 5)
            n.move([(x, y, SAFE_Z)])
            print(f"{PIECES[sq]} {sq}: grip at {z * 1000:.0f} mm -> pos {pos} status {st} "
                  f"({'lifted 3 cm and put back' if held else 'NOT gripped'})", flush=True)
        return
    if mode == "hover":
        x, y = square_xy(sys.argv[2])
        n.move([(p.x, p.y, max(p.z, SAFE_Z)), (x, y, SAFE_Z), (x, y, 0.15)])
        print(f"hovering 15 cm above {sys.argv[2]}", flush=True)
        return
    try:
        results = json.load(open(OUT))
    except (OSError, ValueError):
        results = {}
    todo = sys.argv[2:] or list(PIECES)
    for sq, sym in [(s, PIECES[s]) for s in todo]:
        x, y = square_xy(sq)
        p = n.pose()
        n.move([(p.x, p.y, max(p.z, SAFE_Z)), (x, y, SAFE_Z), (x, y, START_Z)])
        n.gripper(0, 5)
        levels, first = [], None
        z = START_Z
        while z >= MIN_Z:
            n.move([(x, y, z)], vel=0.05)
            pos, st = n.gripper(255)
            n.gripper(0, 5)
            width = 50.0 - pos / 5.1 if st == 2 else None
            levels.append((round(z, 4), pos, st, None if width is None else round(width, 1)))
            print(f"  {sym} {sq} z={z * 1000:5.1f} mm  pos {pos:3d} status {st}  width "
                  f"{'-' if width is None else '%.1f' % width}", flush=True)
            if width is not None and first is None:
                first = z
                if z >= START_Z - 1e-6:
                    print("  contact at the start height - piece taller than expected, stop", flush=True)
                    break
            if first is not None and (z <= first - BODY_MARGIN or (width or 0) > 42):
                break
            z -= STEP
        n.move([(x, y, SAFE_Z)])
        results[sym] = {"square": sq, "first_contact_z": first, "levels": levels}
        json.dump(results, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
