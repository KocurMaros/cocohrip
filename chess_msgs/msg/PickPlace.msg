# One pick-and-place for robot_control, in its game frame: row 0 = robot's
# side ... 7 = human's side, col 0 = human's left (the same frame as
# checkers_msgs).  Values outside 0..7 or fractional are places beside the
# board (storage for captured pieces, reserve for promotions) - robot_control
# maps them with the same two measured corner squares as the board.
float32 pick_row
float32 pick_col
float32 place_row
float32 place_col
# FEN symbol of the carried piece (K Q R B N P / k q r b n p)
string piece
# fingertip (ur5e_robotiq_hande_end) height above the board when gripping [m]
float32 grip_height
# piece width at the grip point [m]; the gripper opens a bit wider before
# descending and closes past it (adaptive grip)
float32 grip_width
# piece centre off its square's centre as the camera sees it, in squares
# (+row towards row 7, +col towards col 7); 0 = centred or unknown
float32 offset_row
float32 offset_col
