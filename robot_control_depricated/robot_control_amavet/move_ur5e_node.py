#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, MoveItErrorCodes, RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import ListControllers, SwitchController
import tf2_ros
import time
from gripper_srv.srv import GripperService

class MoveUr5e(Node):
    def __init__(self):
        super().__init__('move_ur5e_amavet')
        
        # Planning group name
        self.planning_group = "ur_manipulator"
        
        # === MoveIt Action Client ===
        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        self.execute_trajectory_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # === Service Clients ===
        self.gripper_client = self.create_client(GripperService, 'gripper_service')
        self.list_controllers_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        self.switch_controller_client = self.create_client(SwitchController, '/controller_manager/switch_controller')

        # === Subscribers ===
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.current_joint_states = None

        # === State Variables ===
        self.is_robot_moving = False
        
        # Board parameters
        self.square_size = 0.055
        self.board_offset_x = (self.square_size * 12) - 0.0396
        self.board_offset_y = 0.3 - 0.0438

        # Initialization timer
        self._init_timer = self.create_timer(3.0, self.initialization_sequence)

        self.get_logger().info("Robot Control Node (Amavet) starting up...")

    def joint_state_callback(self, msg):
        """Store current joint states."""
        self.current_joint_states = msg

    def activate_controller(self, controller_name='scaled_joint_trajectory_controller'):
        """Activate a controller using the switch_controller service."""
        self.get_logger().info(f"Activating controller: {controller_name}...")
        
        if not self.switch_controller_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Switch controller service not available!")
            return False
        
        request = SwitchController.Request()
        request.activate_controllers = [controller_name]
        request.deactivate_controllers = []
        request.strictness = SwitchController.Request.BEST_EFFORT
        request.activate_asap = True
        request.timeout = rclpy.duration.Duration(seconds=5.0).to_msg()
        
        try:
            future = self.switch_controller_client.call_async(request)
            
            # Manually spin until complete with timeout
            end_time = time.time() + 10.0
            while not future.done() and time.time() < end_time:
                rclpy.spin_once(self, timeout_sec=0.1)
            
            if future.done() and future.result() is not None:
                if future.result().ok:
                    self.get_logger().info(f"Successfully activated {controller_name}")
                    return True
                else:
                    self.get_logger().warn(f"Controller activation returned False (may already be active)")
                    return False
            else:
                self.get_logger().error("Switch controller service call timed out")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Exception while activating controller: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False

    def wait_for_controller_active(self, controller_name='scaled_joint_trajectory_controller', timeout=15.0):
        """Wait for a controller to be active."""
        self.get_logger().info(f"Checking if {controller_name} is active...")
        
        if not self.list_controllers_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Controller manager list service not available!")
            return False
        
        start_time = time.time()
        check_count = 0
        while time.time() - start_time < timeout and rclpy.ok():
            check_count += 1
            request = ListControllers.Request()
            future = self.list_controllers_client.call_async(request)
            
            # Use a shorter timeout and manually spin
            end_time = time.time() + 2.0
            while not future.done() and time.time() < end_time:
                rclpy.spin_once(self, timeout_sec=0.1)
            
            if future.done() and future.result() is not None:
                self.get_logger().info(f"Check {check_count}: Got controller list with {len(future.result().controller)} controllers")
                for controller in future.result().controller:
                    if controller.name == controller_name:
                        self.get_logger().info(f"Found {controller_name}, state: {controller.state}")
                        if controller.state == 'active':
                            self.get_logger().info(f"{controller_name} is ACTIVE!")
                            return True
                        else:
                            self.get_logger().info(f"{controller_name} state: {controller.state}, waiting...")
                            break
                else:
                    # Controller not found in list
                    self.get_logger().warn(f"{controller_name} not found in controller list!")
            else:
                self.get_logger().warn("Failed to get controller list (timeout or no result)")
            
            time.sleep(1.0)
        
        self.get_logger().error(f"Timeout waiting for {controller_name} to be active!")
        return False

    def initialization_sequence(self):
        """Initialize the robot after startup."""
        self._init_timer.cancel()

        # Initialize action clients
        self.get_logger().info("Waiting for action servers...")
        self.execute_trajectory_client.wait_for_server()
        self.get_logger().info("Execute Trajectory server found!")
        
        # Simply wait for the system to stabilize and controller to activate
        # The controller is typically activated by the launch file
        self.get_logger().info("Waiting 8 seconds for system initialization...")
        time.sleep(8.0)
        
        # Now check if controller is active
        self.get_logger().info("Checking controller status...")
        controller_active = False
        
        # Try a few times with simple blocking calls
        for attempt in range(3):
            try:
                request = ListControllers.Request()
                future = self.list_controllers_client.call_async(request)
                
                # Wait for result with timeout
                start = time.time()
                while not future.done() and (time.time() - start) < 3.0:
                    time.sleep(0.1)
                
                if future.done() and future.result() is not None:
                    for controller in future.result().controller:
                        if controller.name == 'scaled_joint_trajectory_controller':
                            if controller.state == 'active':
                                self.get_logger().info("Controller is ACTIVE!")
                                controller_active = True
                                break
                            else:
                                self.get_logger().info(f"Controller state: {controller.state}")
                    
                    if controller_active:
                        break
                else:
                    self.get_logger().warn(f"Attempt {attempt + 1}: Failed to get controller list")
                    
            except Exception as e:
                self.get_logger().error(f"Error checking controller: {e}")
            
            if not controller_active and attempt < 2:
                self.get_logger().info("Retrying in 2 seconds...")
                time.sleep(2.0)
        
        if not controller_active:
            self.get_logger().error("Controller not active after waiting. Please activate manually:")
            self.get_logger().error("ros2 control set_controller_state scaled_joint_trajectory_controller active")
            return
        
        # Wait for joint states
        self.get_logger().info("Waiting for joint states...")
        timeout = time.time() + 5.0
        while self.current_joint_states is None and time.time() < timeout:
            time.sleep(0.1)
        
        if self.current_joint_states is None:
            self.get_logger().error("No joint states received!")
            return

        self.get_logger().info("Got joint states. Performing simple joint move...")
        
        # Simple test: move joint 1 by 0.1 radians
        self.move_joints_simple([0.1, 0, 0, 0, 0, 0], duration=3.0)
        
        # Wait for move to complete
        time.sleep(4.0)

        self.get_logger().info("Testing gripper...")
        self.close_gripper()
        self._gripper_open_timer = self.create_timer(2.0, self.initial_open_gripper)

    def initial_open_gripper(self):
        """Open gripper after initialization."""
        self._gripper_open_timer.cancel()
        self.open_gripper()
        self.get_logger().info("Initialization complete.")

    def move_joints_simple(self, joint_deltas, duration=2.0):
        """
        Move joints by delta values without planning.
        
        Args:
            joint_deltas: List of 6 values to add to current joint positions
            duration: Time to execute the motion (seconds)
        """
        if self.current_joint_states is None:
            self.get_logger().error("No joint states available!")
            return False

        # Get current joint positions
        joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                      'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        current_positions = []
        for joint_name in joint_names:
            try:
                idx = self.current_joint_states.name.index(joint_name)
                current_positions.append(self.current_joint_states.position[idx])
            except ValueError:
                self.get_logger().error(f"Joint {joint_name} not found!")
                return False

        # Calculate target positions
        target_positions = [curr + delta for curr, delta in zip(current_positions, joint_deltas)]
        
        self.get_logger().info(f"Moving joints from {[f'{p:.3f}' for p in current_positions]}")
        self.get_logger().info(f"              to {[f'{p:.3f}' for p in target_positions]}")

        # Create trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names
        
        # Add two points: start and end
        # Start point
        point_start = JointTrajectoryPoint()
        point_start.positions = current_positions
        point_start.time_from_start.sec = 0
        point_start.time_from_start.nanosec = 0
        
        # End point
        point_end = JointTrajectoryPoint()
        point_end.positions = target_positions
        point_end.time_from_start.sec = int(duration)
        point_end.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        
        trajectory.points = [point_start, point_end]

        # Create ExecuteTrajectory goal
        goal = ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory = trajectory
        
        # Send goal
        self.get_logger().info("Sending trajectory to execute...")
        send_goal_future = self.execute_trajectory_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.execute_goal_response_callback)
        
        return True

    def execute_goal_response_callback(self, future):
        """Handle ExecuteTrajectory goal response."""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Trajectory execution rejected!')
                return
            
            self.get_logger().info('Trajectory accepted, executing...')
            self.is_robot_moving = True
            
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.execute_result_callback)
            
        except Exception as e:
            self.get_logger().error(f'Failed to execute trajectory: {e}')
            self.is_robot_moving = False

    def execute_result_callback(self, future):
        """Handle ExecuteTrajectory result."""
        try:
            result = future.result().result
            self.is_robot_moving = False
            
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.get_logger().info('Trajectory execution successful!')
            else:
                self.get_logger().error(f'Trajectory execution failed: {result.error_code.val}')
                
        except Exception as e:
            self.get_logger().error(f'Failed to get execution result: {e}')
            self.is_robot_moving = False

    def move_to_pose(self, pose):
        """Move the robot to a target pose using MoveIt action interface."""
        self.get_logger().info(f'Planning to pose: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})')
        
        goal = MoveGroup.Goal()
        goal.request.group_name = self.planning_group
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1
        
        # Set workspace bounds
        goal.request.workspace_parameters.header.frame_id = "base_link"
        goal.request.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
        goal.request.workspace_parameters.min_corner.x = -1.0
        goal.request.workspace_parameters.min_corner.y = -1.0
        goal.request.workspace_parameters.min_corner.z = -1.0
        goal.request.workspace_parameters.max_corner.x = 1.0
        goal.request.workspace_parameters.max_corner.y = 1.0
        goal.request.workspace_parameters.max_corner.z = 1.0
        
        # Use a simple goal constraint with both position AND orientation
        goal_constraint = Constraints()
        
        # Position constraint with tighter tolerance
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = "base_link"
        position_constraint.header.stamp = self.get_clock().now().to_msg()
        position_constraint.link_name = "tool0"
        position_constraint.weight = 1.0
        
        # Create a small box around the target position
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.001, 0.001, 0.001]  # 1mm tolerance
        
        # Constraint region pose
        constraint_pose = Pose()
        constraint_pose.position = pose.position
        constraint_pose.orientation.w = 1.0
        
        position_constraint.constraint_region.primitives.append(primitive)
        position_constraint.constraint_region.primitive_poses.append(constraint_pose)
        
        # Orientation constraint with exact target orientation
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = "base_link"
        orientation_constraint.header.stamp = self.get_clock().now().to_msg()
        orientation_constraint.link_name = "tool0"
        orientation_constraint.orientation = pose.orientation
        # Tighter tolerances for orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.01
        orientation_constraint.absolute_y_axis_tolerance = 0.01
        orientation_constraint.absolute_z_axis_tolerance = 0.01
        orientation_constraint.weight = 1.0
        
        # Add both constraints
        goal_constraint.position_constraints.append(position_constraint)
        goal_constraint.orientation_constraints.append(orientation_constraint)
        goal.request.goal_constraints.append(goal_constraint)
        
        # Set the planner ID explicitly
        goal.request.planner_id = "RRTConnect"
        
        self.get_logger().info('Sending goal to MoveGroup...')
        try:
            send_goal_future = self.move_group_client.send_goal_async(goal)
            send_goal_future.add_done_callback(self.move_goal_response_callback)
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to plan/execute: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False

    def move_goal_response_callback(self, future):
        """Handle MoveGroup goal response."""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Goal rejected by MoveGroup!')
                return
            
            self.get_logger().info('Goal accepted, waiting for result...')
            self.is_robot_moving = True
            
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.move_result_callback)
            
            # Optional: Add feedback callback to monitor planning
            goal_handle._feedback_callback = self.move_feedback_callback
            
        except Exception as e:
            self.get_logger().error(f'Failed to send goal: {e}')
            self.is_robot_moving = False

    def move_feedback_callback(self, feedback_msg):
        """Handle MoveGroup feedback during planning/execution."""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Planning state: {feedback.state}')

    def move_result_callback(self, future):
        """Handle MoveGroup result."""
        try:
            result = future.result().result
            self.is_robot_moving = False
            
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.get_logger().info('Motion execution successful!')
            else:
                self.get_logger().error(f'Motion failed with error code: {result.error_code.val}')
                
        except Exception as e:
            self.get_logger().error(f'Failed to get result: {e}')
            self.is_robot_moving = False

    def get_current_pose(self, base_frame: str = 'base_link', ee_frame: str = 'tool0', timeout_sec: float = 2.0):
        """Get current end-effector pose using TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_sec)
            )
            
            pose = Pose()
            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z
            pose.orientation = transform.transform.rotation
            
            return True, pose
            
        except Exception as e:
            self.get_logger().warn(f"Cannot transform from {base_frame} to {ee_frame}.")
            return False, None

    def open_gripper(self):
        """Open the gripper."""
        self.get_logger().info("Opening gripper...")
        request = GripperService.Request()
        # 1 to open gripper, 0 to close
        request.position = 1
        request.speed = 5  # Max speed
        request.force = 5  # Max force
            
        future = self.gripper_client.call_async(request)
        future.add_done_callback(self.gripper_response_callback)

    def close_gripper(self):
        """Close the gripper."""
        self.get_logger().info("Closing gripper...")
        request = GripperService.Request()
        # 0 to close gripper
        request.position = 0
        request.speed = 5  # Max speed
        request.force = 5  # Max force

        future = self.gripper_client.call_async(request)
        future.add_done_callback(self.gripper_response_callback)

    def gripper_response_callback(self, future):
        """Handle gripper service response."""
        try:
            response = future.result()
            # Response is a string
            self.get_logger().info(f"Gripper response: {response.response}")
        except Exception as e:
            self.get_logger().error(f"Gripper service call failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MoveUr5e()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
