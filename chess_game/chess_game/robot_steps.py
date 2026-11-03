"""Robot pick/place steps (game.Step) -> geometry for robot_control: game-
frame positions, grip height / width of the physical piece and a travel
height that clears the tallest piece on the board.  No ROS (the node copies
the result into chess_msgs)."""

import os
from dataclasses import dataclass

import chess
import yaml

from chess_game.frames import OffBoardLayout, logical_to_game

DEFAULT_PIECES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config",
                              "pieces.yaml")


@dataclass
class PieceSpec:
    height: float
    grip_height: float
    grip_width: float


def load_pieces(path=None):
    """{'K': PieceSpec, ...} and the travel margin from a pieces.yaml."""
    path = os.path.expanduser(path) if path else _find_pieces_yaml()
    with open(path) as f:
        data = yaml.safe_load(f)
    specs = {k.upper(): PieceSpec(float(v["height"]), float(v["grip_height"]), float(v["grip_width"]))
             for k, v in data["pieces"].items()}
    missing = set("KQRBNP") - set(specs)
    if missing:
        raise ValueError(f"{path}: no entry for {', '.join(sorted(missing))}")
    return specs, float(data.get("travel_margin", 0.025))


def _find_pieces_yaml():
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("chess_game"), "config", "pieces.yaml")
        if os.path.exists(p):
            return p
    except Exception:  # noqa: BLE001 - not installed / no ROS
        pass
    return DEFAULT_PIECES


@dataclass
class StepGeometry:
    pick: tuple            # (row, col) in robot_control's game frame
    place: tuple
    piece: str             # physical FEN symbol
    grip_height: float
    grip_width: float


def place_position(place, human_white, layout):
    if place.kind == "square":
        return tuple(float(v) for v in logical_to_game(place.square, human_white))
    if place.kind == "storage":
        return layout.storage(place.slot)
    raise ValueError(f"unknown place {place}")


def robot_geometry(game, steps, specs, margin, layout=None):
    """(list of StepGeometry, travel height) for `steps` planned on game.board
    (before the move).  The carried piece hangs grip_height below the
    fingertips, so the fingertips travel at tallest piece on the board +
    grip_height of the carried piece + margin."""
    layout = layout or OffBoardLayout()
    out = []
    for s in steps:
        spec = specs[s.symbol.upper()]
        out.append(StepGeometry(place_position(s.pick, game.human_white, layout),
                                place_position(s.place, game.human_white, layout),
                                s.symbol, spec.grip_height, spec.grip_width))
    tallest = max((specs[game.physical_symbol(sq).upper()].height
                   for sq in chess.SQUARES if game.board.piece_at(sq)), default=0.0)
    carried = max((g.grip_height for g in out), default=0.0)
    return out, tallest + carried + margin
