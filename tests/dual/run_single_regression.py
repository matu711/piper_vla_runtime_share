"""Run existing single-arm tests unchanged, with only ROS API mocked."""
import sys
import types
import unittest

from ros_mocks import install, module, Message


def main():
    rospy = install()
    for name in ("loginfo", "logdebug", "logwarn_throttle", "logerr_throttle"):
        setattr(rospy, name, lambda *args, **kwargs: None)
    module("geometry_msgs")
    module("geometry_msgs.msg", PoseStamped=object)
    sys.modules["std_msgs.msg"].Float64 = Message
    suite = unittest.defaultTestLoader.loadTestsFromNames(["test_pbl8_gripper", "test_prompt_sessions", "test_single_runtime"])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # No silently skipped publish-flow tests when dependency imports fail.
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
