"""Fake Hand-E: every close stops on a piece (status 2), every open reaches 0."""
import rclpy
from rclpy.node import Node
from gripper_srv.srv import GripperService


class FakeGripper(Node):
    def __init__(self):
        super().__init__("fake_gripper")
        self.create_service(GripperService, "gripper_service", self.cb)

    def cb(self, req, res):
        closing = req.position > 0
        res.final_position = max(0, req.position - 51) if closing else 0
        res.object_status = 2 if closing else 3
        res.response = "Done"
        self.get_logger().info(f"gripper -> {req.position} (force {req.force}): stop {res.final_position}, status {res.object_status}")
        return res


rclpy.init()
rclpy.spin(FakeGripper())
