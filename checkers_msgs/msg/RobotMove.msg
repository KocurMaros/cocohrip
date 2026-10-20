# RobotMove.msg
bool robot_move_done
# The robot could not grip a piece and gave up the rest of the move (it is
# back home); pick_row / pick_col = that piece's square in game coordinates
bool pick_failed
int32 pick_row
int32 pick_col
