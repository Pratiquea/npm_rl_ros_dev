#!/usr/bin/env python3
"""
Check that the vendored network file still matches the sim's copy byte for byte.

Guards STRUCTURE only. The cone reparametrization changed what the three
continuous outputs mean without touching a line of this file, so a green result
here says nothing about whether the action contract still matches. Skips rather
than fails when the sim repo is absent, so it is safe to run on the robot.

Example:
    /usr/bin/python3 test_net_sync.py
"""
import hashlib
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROS_NET = os.path.join(HERE, "..", "src", "npm_policy", "actor_critic_smdp.py")
SIM_REPO = os.environ.get(
    "NPM_SIM_REPO", os.path.expanduser("~/gits/nonprehensile_object_manipulation"))
SIM_NET = os.path.join(SIM_REPO, "actor_critic_pointer_pointnet_smdp.py")


def sha256(path):
    h = hashlib.sha256()
    with io.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.path.isfile(SIM_NET):
        print("net_sync SKIPPED: %s not found (set NPM_SIM_REPO)" % SIM_NET)
        return

    ours, theirs = sha256(ROS_NET), sha256(SIM_NET)
    if ours != theirs:
        diff = subprocess.Popen(["diff", "-u", SIM_NET, ROS_NET],
                                stdout=subprocess.PIPE).communicate()[0]
        sys.stderr.write(diff.decode("utf-8", "replace"))
        raise AssertionError(
            "network file drifted from the sim.\n"
            "  ros: %s %s\n  sim: %s %s\n"
            "Re-sync deliberately once you have checked the diff above:\n"
            "  cp %s %s\n"
            "and confirm any checkpoint in use still loads afterwards."
            % (ROS_NET, ours, SIM_NET, theirs, SIM_NET, ROS_NET))

    print("net_sync OK: actor_critic_smdp.py matches the sim (sha256 %s)" % ours[:12])


if __name__ == "__main__":
    main()
