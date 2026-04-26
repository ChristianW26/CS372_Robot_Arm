"""
Interactive terminal joint control for SO100/SO101 follower arm.

Usage:
    python so100_joint_control.py

Commands (once running):
    <joint> <degrees>   — move a joint to a position in degrees (gripper: 0-100%)
    r                   — read and print current positions
    home                — move all joints to 0 (gripper to 50%)
    q                   — quit

Joint names:
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
"""

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/ttyACM0"
ARM_ID = "CORL_follower_arm"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

HOME = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}


def print_help():
    print("\nCommands:")
    print("  <joint> <value>  — set joint position (degrees; gripper 0-100%)")
    print("  r                — read current joint positions")
    print("  home             — move all joints to home position")
    print("  q                — quit")
    print(f"\nJoints: {', '.join(JOINTS)}\n")


def read_and_print(robot):
    obs = robot.get_observation()
    print("\nCurrent positions:")
    for k, v in obs.items():
        if k.endswith(".pos"):
            joint = k.removesuffix(".pos")
            unit = "%" if joint == "gripper" else "deg"
            print(f"  {joint:<20} {v:+.2f} {unit}")
    print()


def main():
    print(f"Connecting to SO101 on {PORT} (id={ARM_ID})...")
    config = SO101FollowerConfig(port=PORT, id=ARM_ID)
    robot = SO101Follower(config)
    robot.connect(calibrate=False)
    print("Connected.\n")

    print_help()
    read_and_print(robot)

    # Keep the last known action so we only change one joint at a time.
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
                read_and_print(robot)
                continue

            if raw.lower() == "home":
                robot.send_action(HOME)
                current = dict(HOME)
                print("Moved to home.")
                read_and_print(robot)
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
