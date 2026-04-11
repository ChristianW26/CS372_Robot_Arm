"""
SO100/SO101 sim2real policy deployment — interactive mode with live ArUco cube tracking.

Loads a trained actor network (ckpt_*.pt, PPO checkpoint) and lets you
step through the observe → infer → execute loop manually.

Usage:
    python so100_policy_deploy.py [--ckpt PATH] [--dry-run] [--scale 0.5] [--cam INDEX]

Terminal commands (once running):
    obs        — read robot state, run model, print proposed action
    y / yes    — execute the last proposed action
    n / no     — discard the last proposed action (do nothing)
    run [N]    — auto obs+execute N steps at ~60 Hz (default 10); press Ctrl-C to abort
    q / quit   — exit

Options:
    --ckpt     Path to checkpoint file (default: ../ckpt_2401.pt)
    --dry-run  Compute and print actions but do NOT send to robot
    --scale    Safety multiplier on action deltas, e.g. 0.5 = half speed (default: 1.0)
    --cam      OpenCV camera index for ArUco detection (default: 0)

Cube pose is detected live from an ArUco 4×4 marker (25 mm) on the top face of the cube.
If detection fails in a given step the last known pose is reused automatically.
Goal is set once at startup as: initial_cube_pos + [0, 0, GOAL_Z_OFFSET].
"""

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as ScipyRotation

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.rotation import Rotation

sys.path.insert(0, "/home/cpsl_thor1/Desktop/CORL")
from cube_pose_from_aruco import (
    get_cube_pose_from_image,
    MARKER_DICT, MARKER_ID, MARKER_LENGTH, CUBE_HALF_SIZE,
)

# ── Hardware config ───────────────────────────────────────────────────────────
PORT      = "/dev/ttyACM0"
ARM_ID    = "CORL_follower_arm"
URDF_PATH = "/home/cpsl_thor1/Desktop/CORL/SO-ARM100/Simulation/SO101"
DEFAULT_CKPT = "/home/cpsl_thor1/Desktop/CORL/ckpt_2401.pt"

# ── Control config ────────────────────────────────────────────────────────────
CONTROL_DT = 1.0 / 50.0  # used only for velocity finite-difference

FK_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Action delta scales: policy output in [-1,1] → radians
# arm joints: ±0.05 rad,  gripper: ±0.2 rad  (from so100_sim2real_specs.md)
ACTION_SCALES = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.2], dtype=np.float32)

# ── Camera → Robot-base transform ─────────────────────────────────────────────
# CAM_T: camera optical centre in robot base frame (metres).
CAM_T = np.array([0.0784, 0.2697, 0.3299], dtype=np.float64)

# CAM_R_BASE: 3×3 matrix R such that  p_base = R @ p_cam + CAM_T.
#
# Camera orientation (physically measured):
#   • 50° panned LEFT  of the robot's forward (+X) direction  → yaw  = +50°  around base Z
#   • 35° tilted  DOWN from horizontal                         → pitch = −35°  around camera Y
#
# OpenCV camera frame convention:
#   cam-Z  →  optical axis (into scene)
#   cam-X  →  right in image
#   cam-Y  →  down  in image
#
# We build R as:
#   R = R_pan_tilt  @  R_optical_to_base_aligned
# where R_optical_to_base_aligned converts from the OpenCV frame to a
# "camera pointing straight ahead, level" base-aligned frame, and
# R_pan_tilt applies the measured pan/tilt to that aligned frame.
#
# If the result produces wrong cube positions, verify with a known point
# and adjust the angles below.
_R_optical = np.array([          # OpenCV cam → base-aligned forward-facing camera
    [ 0,  0,  1],                 # cam-Z (optical)  → base +X (forward)
    [-1,  0,  0],                 # cam-X (right)     → base -Y (right)
    [ 0, -1,  0],                 # cam-Y (down)      → base -Z (down)
], dtype=np.float64)

_R_pan_tilt = ScipyRotation.from_euler(
    'ZY',                         # intrinsic: first yaw around Z, then pitch around new Y
    [-50.0, 27.0],                # yaw: −50° = right of forward (flip sign if cube x/y are mirrored)
                                  # pitch: +27° (scipy R_y: positive = nose down)
    degrees=True,
).as_matrix()

CAM_R_BASE = _R_pan_tilt @ _R_optical

# Z offset: robot base is this many metres BELOW the table surface.
# Subtract from all z coordinates so the policy sees z values matching sim training
# (sim cube center z ≈ 0.0125 m above table).
# Measure: put cube on table, run scan, read base z → Z_TABLE_OFFSET = that z - 0.0125
Z_TABLE_OFFSET = 0.057   # metres  (0.069 detected − 0.0125 cube half-size ≈ 0.057)

# Additive correction applied to every ArUco detection in base frame.
# Recompute with the 'calibrate' command in debug_pose.py whenever the camera moves.
CAM_POS_CORRECTION = np.array([-0.047, +0.106, -0.023])

# Goal is placed this many metres above the cube's detected starting Z position (post-offset).
GOAL_Z_OFFSET = 0.05   # 5 cm lift

# Gripper unit conversion: 0% = 0 rad (closed), 100% = 1.8 rad (open)
GRIPPER_PCT_TO_RAD = 1.8 / 100.0

# Grasp heuristic: gripper % below this → is_grasped = 1
GRASP_PCT_THRESHOLD = 20.0


# ── Camera transform ─────────────────────────────────────────────────────────

def cam_to_base(
    pos_cam: np.ndarray,
    quat_wxyz_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform a pose from camera frame to robot base frame.

      p_base  = CAM_R_BASE @ p_cam + CAM_T
      q_base  = R_base_cam * R_marker   (rotation composition)

    pos_cam       : (3,) position in camera frame, metres
    quat_wxyz_cam : (4,) quaternion [w, x, y, z] in camera frame

    Returns (pos_base, quat_wxyz_base).
    """
    pos_base = CAM_R_BASE @ pos_cam.astype(np.float64) + CAM_T
    pos_base[2] -= Z_TABLE_OFFSET
    pos_base    += CAM_POS_CORRECTION

    # Compose rotations: base←cam  then  cam←marker
    R_base_cam = ScipyRotation.from_matrix(CAM_R_BASE)
    w, x, y, z = quat_wxyz_cam.astype(np.float64)
    R_marker   = ScipyRotation.from_quat([x, y, z, w])   # scipy wants [x,y,z,w]
    R_result   = R_base_cam * R_marker
    xyzw       = R_result.as_quat()                        # scipy returns [x,y,z,w]
    quat_base  = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
    if quat_base[0] < 0:                                   # keep qw positive (q and -q are identical rotations)
        quat_base = -quat_base

    return pos_base.astype(np.float32), quat_base


# ── ArUco cube tracker ────────────────────────────────────────────────────────

class ArucoTracker:
    """
    Wraps an OpenCV camera + ArUco detection.

    Calls get_cube_pose_from_image() on each frame, transforms the result to
    robot base frame, and caches the last successful detection so callers can
    always get a pose even when the marker is temporarily occluded.
    """

    def __init__(self, cam_idx: int):
        import cv2
        from aruco_markers.detect import load_camera_parameters
        from aruco_markers.camera import cvCamera

        self._cam   = cvCamera(cam_idx)
        self._cammat, self._dist = load_camera_parameters(self._cam.name)

        # Last known cube pose in robot base frame
        self._last_pos:     np.ndarray | None = None
        self._last_quat:    np.ndarray | None = None
        self._last_pos_cam: np.ndarray | None = None  # raw camera frame (debug)
        self._n_miss = 0   # consecutive detection failures

        print(f"ArUco tracker: camera '{self._cam.name}' index={cam_idx}")

    def update(self) -> bool:
        """
        Grab one camera frame, try to detect the cube marker.
        Returns True if detection succeeded (pose was refreshed).
        If detection fails, the cached pose is unchanged.
        """
        img    = self._cam.read()
        result = get_cube_pose_from_image(img, self._cammat, self._dist)

        if result is None:
            self._n_miss += 1
            return False

        pos_cam, quat_cam = result          # camera frame, quat [w,x,y,z]
        self._last_pos_cam = pos_cam.copy() # keep raw cam-frame reading for debug
        pos_base, quat_base = cam_to_base(pos_cam, quat_cam)

        self._last_pos  = pos_base
        self._last_quat = quat_base
        self._n_miss    = 0
        return True

    @property
    def pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (pos, quat [w,x,y,z]) in base frame, or None if never detected."""
        if self._last_pos is None:
            return None
        return self._last_pos.copy(), self._last_quat.copy()

    @property
    def pose_cam(self) -> np.ndarray | None:
        """Raw camera-frame position of last detection (for debugging the transform)."""
        return self._last_pos_cam.copy() if self._last_pos_cam is not None else None

    @property
    def consecutive_misses(self) -> int:
        return self._n_miss

    def close(self) -> None:
        self._cam.close()


# ── Actor network ─────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """5-layer MLP matching the CleanRL PPO actor in the checkpoint."""

    def __init__(self, obs_dim: int = 36, act_dim: int = 6, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


def load_actor(ckpt_path: str) -> Actor:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor_sd = {"net." + k.removeprefix("actor_mean."): v
                for k, v in sd.items() if k.startswith("actor_mean.")}
    actor = Actor()
    actor.load_state_dict(actor_sd)
    actor.eval()
    print(f"Loaded actor from {ckpt_path}")
    return actor


# ── Kinematics ────────────────────────────────────────────────────────────────

def load_kinematics():
    try:
        kin = RobotKinematics(urdf_path=URDF_PATH, target_frame_name="gripper_frame_link")
        print(f"FK solver loaded from {URDF_PATH}")
        return kin
    except FileNotFoundError:
        print(f"[warn] URDF not found at '{URDF_PATH}/robot.urdf'. TCP pose will be zeroed.")
    except ImportError:
        print("[warn] 'placo' not installed. TCP pose will be zeroed.")
    return None


def compute_tcp_pose(
    arm_qpos_deg: np.ndarray,
    kinematics,
) -> tuple[np.ndarray, np.ndarray]:
    """FK for the 5 arm joints → (tcp_pos [x,y,z], tcp_quat [qw,qx,qy,qz]).

    Coordinate frame (robot base = origin, matches URDF world frame):
        X — forward (direction the arm reaches out)
        Y — left/right side
        Z — up
    All positions in metres. Quaternion order: [qw, qx, qy, qz].

    arm_qpos_deg: degrees, matching what the robot API reports.
    kinematics.forward_kinematics() expects degrees and converts internally.
    """
    T   = kinematics.forward_kinematics(arm_qpos_deg.astype(np.float64))
    pos = T[:3, 3].astype(np.float32)
    rot = Rotation.from_matrix(T[:3, :3])
    qx, qy, qz, qw = rot._quat
    return pos, np.array([qw, qx, qy, qz], dtype=np.float32)


# ── Observation builder ───────────────────────────────────────────────────────

def build_observation(
    robot_obs: dict,
    prev_qpos_rad: np.ndarray | None,
    tcp_pos: np.ndarray,
    tcp_quat: np.ndarray,
    cube_pos: np.ndarray,
    cube_quat: np.ndarray,
    goal_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the 36-dim observation vector for the trained policy.

    Returns (obs [36,], qpos_rad [6,]).
    """
    arm_deg     = np.array([robot_obs[f"{n}.pos"] for n in FK_MOTOR_NAMES], dtype=np.float32)
    arm_rad     = np.deg2rad(arm_deg)
    gripper_pct = float(robot_obs["gripper.pos"])
    gripper_rad = np.float32(gripper_pct * GRIPPER_PCT_TO_RAD)
    qpos        = np.append(arm_rad, gripper_rad)

    qvel = (qpos - prev_qpos_rad) / CONTROL_DT if prev_qpos_rad is not None else np.zeros(6, dtype=np.float32)

    is_grasped  = np.float32(1.0 if gripper_pct < GRASP_PCT_THRESHOLD else 0.0)
    tcp_to_obj  = cube_pos - tcp_pos
    obj_to_goal = goal_pos - cube_pos

    obs = np.concatenate([
        qpos,           # 0-5   joint positions (rad)
        qvel,           # 6-11  joint velocities (rad/s)
        [is_grasped],   # 12    grasp state
        tcp_pos,        # 13-15 TCP position (m)
        tcp_quat,       # 16-19 TCP quaternion [qw,qx,qy,qz]
        goal_pos,       # 20-22 goal position (m)
        cube_pos,       # 23-25 cube position (m)
        cube_quat,      # 26-29 cube quaternion [qw,qx,qy,qz]
        tcp_to_obj,     # 30-32 TCP→cube vector (m)
        obj_to_goal,    # 33-35 cube→goal vector (m)
    ]).astype(np.float32)

    assert obs.shape == (36,), f"obs shape error: {obs.shape}"
    return obs, qpos


# ── Action helpers ────────────────────────────────────────────────────────────

def action_to_command(
    action_norm: np.ndarray,
    current_qpos_rad: np.ndarray,
    scale: float = 1.0,
) -> tuple[dict, np.ndarray]:
    """
    Denormalize policy output and produce a robot command dict.

    Returns (cmd dict for send_action, new_qpos_rad after applying delta).
    """
    deltas       = action_norm * ACTION_SCALES * scale
    new_qpos_rad = current_qpos_rad + deltas

    cmd = {f"{name}.pos": float(np.rad2deg(new_qpos_rad[i]))
           for i, name in enumerate(FK_MOTOR_NAMES)}
    gripper_pct      = float(new_qpos_rad[5] / GRIPPER_PCT_TO_RAD)
    cmd["gripper.pos"] = float(np.clip(gripper_pct, 0.0, 100.0))

    return cmd, new_qpos_rad


# ── Interactive helpers ───────────────────────────────────────────────────────

def print_help() -> None:
    print("\nCommands:")
    print("  scan       — continuously print cube position (cam + base frame); press Ctrl-C to stop")
    print("  obs        — read robot state, run model, show proposed action")
    print("  y / yes    — execute the last proposed action")
    print("  n / no     — discard the last proposed action")
    print("  run [N]    — auto obs+execute N steps at ~60 Hz (default 10); press Ctrl-C to abort")
    print("  q / quit   — exit\n")


def prompt_vec(label: str, default: list[float]) -> np.ndarray:
    print(f"  {label} [{' '.join(f'{v:.4f}' for v in default)}]: ", end="", flush=True)
    raw = input().strip()
    return np.array([float(v) for v in raw.split()], dtype=np.float32) if raw else np.array(default, dtype=np.float32)


def print_obs(obs: np.ndarray, tcp_pos: np.ndarray, tcp_quat: np.ndarray) -> None:
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    print("\n── Observation ──────────────────────────────────")
    print("  Joint positions (rad):")
    for i, n in enumerate(names):
        print(f"    [{i}] {n:<20} {obs[i]:+.4f}")
    print("  Joint velocities (rad/s):")
    for i, n in enumerate(names):
        print(f"    [{i+6}] {n+'_vel':<20} {obs[i+6]:+.4f}")
    print(f"  [12] is_grasped           {obs[12]:.1f}")
    print(f"  TCP pos  (m)  : [{tcp_pos[0]:+.4f}, {tcp_pos[1]:+.4f}, {tcp_pos[2]:+.4f}]")
    print(f"  TCP quat      : [{tcp_quat[0]:+.4f}, {tcp_quat[1]:+.4f}, {tcp_quat[2]:+.4f}, {tcp_quat[3]:+.4f}]")
    print(f"  goal_pos      : [{obs[20]:+.4f}, {obs[21]:+.4f}, {obs[22]:+.4f}]")
    print(f"  cube_pos      : [{obs[23]:+.4f}, {obs[24]:+.4f}, {obs[25]:+.4f}]")
    print(f"  tcp→cube      : [{obs[30]:+.4f}, {obs[31]:+.4f}, {obs[32]:+.4f}]")
    print(f"  cube→goal     : [{obs[33]:+.4f}, {obs[34]:+.4f}, {obs[35]:+.4f}]")


def print_action(action_norm: np.ndarray, scale: float) -> None:
    names   = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    deltas  = action_norm * ACTION_SCALES * scale
    print("\n── Proposed Action ──────────────────────────────")
    print(f"  {'Joint':<20} {'norm [-1,1]':>12}  {'delta (rad)':>12}")
    for i, n in enumerate(names):
        print(f"  {n:<20} {action_norm[i]:+12.4f}  {deltas[i]:+12.4f}")
    if scale != 1.0:
        print(f"  (scale={scale})")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt",    default=DEFAULT_CKPT, help="Path to .pt checkpoint")
    p.add_argument("--dry-run", action="store_true",  help="Do not send commands to robot")
    p.add_argument("--scale",   type=float, default=1.0, help="Action delta scale factor")
    p.add_argument("--cam",     type=int,   default=0,   help="OpenCV camera index for ArUco (default: 0)")
    return p.parse_args()


def obs_and_infer(
    robot,
    kinematics,
    actor: Actor,
    prev_qpos_rad: np.ndarray | None,
    cube_pos: np.ndarray,
    cube_quat: np.ndarray,
    goal_pos: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    One observe+infer cycle. Returns (obs, tcp_pos, action_norm, cmd, new_qpos_rad).
    Caller is responsible for updating prev_qpos_rad afterwards.
    """
    if robot is not None:
        robot_obs = robot.get_observation()
    else:
        robot_obs = {f"{n}.pos": 0.0 for n in FK_MOTOR_NAMES}
        robot_obs["gripper.pos"] = 50.0

    arm_deg = np.array([robot_obs[f"{n}.pos"] for n in FK_MOTOR_NAMES], dtype=np.float32)
    if kinematics is not None:
        tcp_pos, tcp_quat = compute_tcp_pose(arm_deg, kinematics)
    else:
        tcp_pos  = np.zeros(3,  dtype=np.float32)
        tcp_quat = np.array([1., 0., 0., 0.], dtype=np.float32)

    obs, current_qpos_rad = build_observation(
        robot_obs, prev_qpos_rad,
        tcp_pos, tcp_quat,
        cube_pos, cube_quat, goal_pos,
    )

    with torch.no_grad():
        action_norm = actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

    cmd, new_qpos_rad = action_to_command(action_norm, current_qpos_rad, scale)
    return obs, tcp_pos, tcp_quat, action_norm, cmd, new_qpos_rad


def main() -> None:
    args = parse_args()

    actor      = load_actor(args.ckpt)
    kinematics = load_kinematics()

    # ── ArUco tracker setup ───────────────────────────────────────────────────
    tracker = ArucoTracker(cam_idx=args.cam)

    print("\n=== Detecting initial cube pose (point camera at cube) ===")
    cube_pos  = None
    cube_quat = None
    for attempt in range(30):   # up to 3 seconds at ~10 fps
        if tracker.update():
            cube_pos, cube_quat = tracker.pose
            print(f"  Detected: pos={cube_pos}  quat={cube_quat}")
            break
        time.sleep(0.1)
        print(f"  Waiting for marker... ({attempt+1}/30)", end="\r")

    if cube_pos is None:
        print("\n[warn] Marker not detected after 30 attempts.")
        print("       Falling back to manual entry (press Enter for defaults).")
        cube_pos  = prompt_vec("Cube position  (x y z, meters)  ", [0.25, 0.00, 0.0125])
        cube_quat = prompt_vec("Cube quaternion (qw qx qy qz)   ", [1.0,  0.00, 0.00, 0.00])

    # Goal = detected starting position + small lift in Z
    goal_pos = cube_pos.copy()
    goal_pos[2] += GOAL_Z_OFFSET
    print(f"  Goal set to: {goal_pos}  (cube z + {GOAL_Z_OFFSET} m)")

    if not args.dry_run:
        print(f"\nConnecting to SO101 on {PORT} (id={ARM_ID}) ...")
        config = SO101FollowerConfig(port=PORT, id=ARM_ID)
        robot  = SO101Follower(config)
        robot.connect(calibrate=False)
        print("Connected.")
    else:
        robot = None
        print("\n[dry-run] Robot not connected.")

    print_help()

    prev_qpos_rad: np.ndarray | None  = None
    pending_cmd:   dict | None        = None   # last computed command, waiting for y/n
    pending_qpos:  np.ndarray | None  = None   # qpos that would result from pending_cmd

    try:
        while True:
            try:
                raw = input(">> ").strip().lower()
            except EOFError:
                break

            if not raw:
                continue

            # ── quit ─────────────────────────────────────────────────────────
            if raw in ("q", "quit"):
                break

            # ── scan: live cube position readout for transform debugging ─────
            if raw == "scan":
                print("Scanning — move the cube around. Press Ctrl-C to stop.\n")
                print(f"  {'cam x':>8}  {'cam y':>8}  {'cam z':>8}    {'base x':>8}  {'base y':>8}  {'base z':>8}")
                try:
                    while True:
                        detected = tracker.update()
                        if detected:
                            c = tracker.pose_cam
                            b, _ = tracker.pose
                            print(f"  {c[0]:+8.4f}  {c[1]:+8.4f}  {c[2]:+8.4f}    {b[0]:+8.4f}  {b[1]:+8.4f}  {b[2]:+8.4f}")
                        else:
                            print("  (no marker)")
                        time.sleep(0.2)
                except KeyboardInterrupt:
                    print("\nScan stopped.")
                continue

            # ── capture observation & run model ───────────────────────────────
            if raw == "obs":
                obs, tcp_pos, tcp_quat, action_norm, pending_cmd, pending_qpos = obs_and_infer(
                    robot, kinematics, actor, prev_qpos_rad,
                    cube_pos, cube_quat, goal_pos, args.scale,
                )
                prev_qpos_rad = pending_qpos - action_norm * ACTION_SCALES * args.scale  # restore pre-action qpos for vel calc
                print_obs(obs, tcp_pos, tcp_quat)
                print_action(action_norm, args.scale)
                print("\nExecute this action? [y/n]")
                continue

            # ── auto run N steps at 10 Hz ─────────────────────────────────────
            if raw.startswith("run"):
                parts = raw.split()
                try:
                    n_steps = int(parts[1]) if len(parts) > 1 else 10
                except ValueError:
                    print("Usage: run [N]  (N must be an integer)")
                    continue

                print(f"Running {n_steps} steps at ~60 Hz — press Ctrl-C to abort.\n")
                run_dt = 1.0 / 60.0
                try:
                    for step in range(n_steps):
                        t0 = time.perf_counter()

                        # ── Refresh cube pose from ArUco; fall back on miss ───
                        detected = tracker.update()
                        if detected:
                            cube_pos, cube_quat = tracker.pose
                        miss_str = f" [miss×{tracker.consecutive_misses}]" if not detected else ""

                        obs, tcp_pos, tcp_quat, action_norm, cmd, new_qpos_rad = obs_and_infer(
                            robot, kinematics, actor, prev_qpos_rad,
                            cube_pos, cube_quat, goal_pos, args.scale,
                        )
                        prev_qpos_rad = new_qpos_rad.copy()
                        print(f"  step {step+1:02d}/{n_steps}"
                              f"  cube: [{cube_pos[0]:+.3f},{cube_pos[1]:+.3f},{cube_pos[2]:+.3f}]{miss_str}"
                              f"  tcp: [{tcp_pos[0]:+.3f},{tcp_pos[1]:+.3f},{tcp_pos[2]:+.3f}]"
                              f"  act: {np.array2string(action_norm, precision=2, suppress_small=True)}")
                        if robot is not None:
                            robot.send_action(cmd)
                        else:
                            print(f"           [dry-run] cmd: { {k: round(v,3) for k,v in cmd.items()} }")
                        elapsed = time.perf_counter() - t0
                        sleep_t = run_dt - elapsed
                        if sleep_t > 0:
                            time.sleep(sleep_t)
                except KeyboardInterrupt:
                    print("\nRun aborted.")
                pending_cmd  = None
                pending_qpos = None
                print("Done.")
                continue

            # ── accept ───────────────────────────────────────────────────────
            if raw in ("y", "yes"):
                if pending_cmd is None:
                    print("No action pending. Run 'obs' first.")
                    continue
                if robot is not None:
                    robot.send_action(pending_cmd)
                    print("Action sent.")
                else:
                    print(f"[dry-run] Would send: { {k: round(v,3) for k,v in pending_cmd.items()} }")
                prev_qpos_rad = pending_qpos.copy()
                pending_cmd   = None
                pending_qpos  = None
                continue

            # ── reject ───────────────────────────────────────────────────────
            if raw in ("n", "no"):
                if pending_cmd is None:
                    print("No action pending.")
                    continue
                print("Action discarded.")
                pending_cmd  = None
                pending_qpos = None
                continue

            print(f"Unknown command '{raw}'. ", end="")
            print_help()

    finally:
        tracker.close()
        if robot is not None:
            robot.disconnect()
            print("Disconnected.")


if __name__ == "__main__":
    main()
