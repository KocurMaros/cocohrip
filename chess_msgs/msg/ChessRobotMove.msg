# A robot chess move: executed step by step, in order (captured piece
# off first, castling = king then rook; a promoting pawn just moves - the
# new piece exists only in the game).
# robot_control answers with checkers_msgs/RobotMove on /robot_move_topic.
string uci
string san
PickPlace[] steps
# fingertip height for moving between squares [m]: above the tallest piece
# on the board plus the length of the carried piece hanging below the fingers
float32 travel_height
