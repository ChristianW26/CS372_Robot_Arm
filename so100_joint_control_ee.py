"""
Interactive terminal joint control for SO100/SO101 follower arm — with TCP pose display.

Usage:
    python so100_joint_control_ee.py

Requires the SO101 URDF (see URDF_PATH below). Download from:
    https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
and place it at the path set in URDF_PATH.

Commands (once running):
    <joint> <degrees>   — move a joint to a position in degrees (gripper: 0-100%)
    r                   — read and print current positions + TCP pose
    home                — move all joints to 0 (gripper to 50%)
    q                   — quit

Joint names:
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

TCP pose format printed: [x, y, z, qw, qx, qy, qz]  (position in meters)
"""

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.rotation import Rotation

PORT = "/dev/ttyACM0"
ARM_ID = "CORL_follower_arm"

# placo.RobotWrapper expects a *directory* containing robot.urdf + its mesh assets.
# Clone the SO-ARM100 repo and point this to the SO101 simulation folder:
#   git clone --depth=1 https://github.com/TheRobotStudio/SO-ARM100.git /home/cpsl_thor1/Desktop/CORL/SO-ARM100
# Then rename the URDF to robot.urdf:
#   cp SO-ARM100/Simulation/SO101/so101_new_calib.urdf SO-ARM100/Simulation/SO101/robot.urdf
URDF_PATH = "/home/cpsl_thor1/Desktop/CORL/SO-ARM100/Simulation/SO101"

# Arm joints fed into FK (gripper is excluded — not part of the kinematic chain).
FK_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

HOME = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}


def load_kinematics():
    """Load the FK solver. Returns None with a warning if URDF is missing or placo not installed."""
    try:
        kin = RobotKinematics(urdf_path=URDF_PATH, target_frame_name="gripper_frame_link")
        print(f"FK solver loaded from {URDF_PATH}")
        return kin
    except FileNotFoundError:
        print(f"[warn] URDF not found at '{URDF_PATH}/robot.urdf'. TCP pose will not be shown.")
        print(f"       Run: mkdir -p SO101 && wget -O SO101/robot.urdf 'https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf'")
    except ImportError:
        print("[warn] 'placo' not installed. TCP pose will not be shown.")
        print("       Install via: pip install lerobot[kinematics]")
    return None


def compute_tcp_pose(obs, kinematics):
    """
    Compute TCP pose from joint observation dict.
    Returns [x, y, z, qw, qx, qy, qz] with position in meters, or None on error.
    """
    q = np.array([obs[f"{n}.pos"] for n in FK_MOTOR_NAMES], dtype=float)
    T = kinematics.forward_kinematics(q)          # 4x4 homogeneous transform
    pos = T[:3, 3]                                 # [x, y, z] in meters
    rot = Rotation.from_matrix(T[:3, :3])
    qx, qy, qz, qw = rot._quat                    # stored internally as [x, y, z, w]
    return [pos[0], pos[1], pos[2], qw, qx, qy, qz]


def print_help():
    print("\nCommands:")
    print("  <joint> <value>  — set joint position (degrees; gripper 0-100%)")
    print("  r                — read current joint positions + TCP pose")
    print("  home             — move all joints to home position")
    print("  q                — quit")
    print(f"\nJoints: {', '.join(JOINTS)}\n")


def read_and_print(robot, kinematics):
    obs = robot.get_observation()
    print("\nCurrent positions:")
    for k, v in obs.items():
        if k.endswith(".pos"):
            joint = k.removesuffix(".pos")
            unit = "%" if joint == "gripper" else "deg"
            print(f"  {joint:<20} {v:+.2f} {unit}")

    if kinematics is not None:
        tcp = compute_tcp_pose(obs, kinematics)
        x, y, z, qw, qx, qy, qz = tcp
        print(f"\n  TCP pose  [x, y, z, qw, qx, qy, qz]:")
        print(f"    pos  (m):  [{x:+.4f},  {y:+.4f},  {z:+.4f}]")
        print(f"    quat    :  [{qw:+.4f},  {qx:+.4f},  {qy:+.4f},  {qz:+.4f}]")
    print()


def main():
    print(f"Connecting to SO101 on {PORT} (id={ARM_ID})...")
    config = SO101FollowerConfig(port=PORT, id=ARM_ID)
    robot = SO101Follower(config)
    robot.connect(calibrate=False)
    print("Connected.\n")

    kinematics = load_kinematics()

    print_help()
    read_and_print(robot, kinematics)

    obs = robot.get_observation()
    current = {k: v for k, v in obs.items() if k.endswith(".pos")}

    try:
        while True:
            try:
                raw = input(">> ").strip()
            except EOFError:
                break

            if not raw:
                continue

            if raw.lower() == "q":
                break

            if raw.lower() == "r":
                read_and_print(robot, kinematics)
                continue

            if raw.lower() == "home":
                robot.send_action(HOME)
                current = dict(HOME)
                print("Moved to home.")
                read_and_print(robot, kinematics)
                continue

            parts = raw.split()
            if len(parts) != 2:
                print("Usage: <joint> <value>  (or r / home / q)")
                continue

            joint, val_str = parts
            if joint not in JOINTS:
                print(f"Unknown joint '{joint}'. Valid: {', '.join(JOINTS)}")
                continue

            try:
                val = float(val_str)
            except ValueError:
                print(f"Invalid value '{val_str}'. Must be a number.")
                continue

            key = f"{joint}.pos"
            current[key] = val
            robot.send_action(current)
            unit = "%" if joint == "gripper" else "deg"
            print(f"  -> {joint} = {val:+.2f} {unit}")

    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
