# Piece.msg
int32 row
int32 col
string color
bool king
# Piece centre relative to its square's centre, in squares (game frame:
# +row towards row 7, +col towards col 7); 0 = centred or unknown
float32 offset_row
float32 offset_col
