"""Publish one chess move (capture + castling style steps) and wait for done."""
import sys, time
import rclpy
from rclpy.node import Node
from chess_msgs.msg import ChessRobotMove, PickPlace
from checkers_msgs.msg import RobotMove

rclpy.init()
node = Node("send_chess_move")
done = []
node.create_subscription(RobotMove, "/robot_move_topic", lambda m: done.append(m), 10)
pub = node.create_publisher(ChessRobotMove, "/chess/robot_move", 10)
msg = ChessRobotMove(uci="e8g8", san="O-O", travel_height=0.14)
# captured piece to the left storage, then king e8->g8 and rook h8->f8 (human white)
msg.steps = [
    PickPlace(pick_row=2.0, pick_col=3.0, place_row=0.0, place_col=-1.8, piece="P", grip_height=0.018, grip_width=0.019),
    PickPlace(pick_row=0.0, pick_col=4.0, place_row=0.0, place_col=6.0, piece="k", grip_height=0.030, grip_width=0.024),
    PickPlace(pick_row=0.0, pick_col=7.0, place_row=0.0, place_col=5.0, piece="r", grip_height=0.022, grip_width=0.024),
]
t0 = time.time()
while pub.get_subscription_count() == 0 and time.time() - t0 < 30:
    rclpy.spin_once(node, timeout_sec=0.2)
print("robot_control subscribed:", pub.get_subscription_count() > 0, flush=True)
pub.publish(msg)
t0 = time.time()
while not done and time.time() - t0 < float(sys.argv[1] if len(sys.argv) > 1 else 240):
    rclpy.spin_once(node, timeout_sec=0.5)
print("DONE" if done else "TIMEOUT", f"after {time.time() - t0:.0f} s", done[0] if done else "", flush=True)
