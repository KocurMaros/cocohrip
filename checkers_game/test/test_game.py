"""Hardware-free tests: inferring the human's move from camera grids,
huff handling and the robot's turn.

Run:  cd checkers_game && python3 -m pytest test/test_game.py -q
"""
import random

from checkers_game.checkers.game import BLACK, HUMAN, ROBOT, WHITE, Game
from checkers_game.checkers.rules import GameState, Move


def grid_after(state, move):
    return state.apply(move).to_grid()


def new_game(human_first=True):
    g = Game(human_first=human_first)
    g.start_standard()
    return g


def test_standard_start_detection():
    grid = GameState.initial(1).to_grid()
    ok, _, parity = Game.check_standard_start(grid)
    assert ok and parity == 1
    grid[3][0] = 1
    ok, msg, _ = Game.check_standard_start(grid)
    assert not ok and "white=13" in msg


def test_robot_moves_first_by_default():
    g = Game()
    g.start_standard()
    assert g.turn == ROBOT
    plan = g.plan_robot_turn(level=2, rng=random.Random(0))
    assert plan.move in g.state.legal_moves()
    g.apply_robot(plan)
    assert g.turn == HUMAN


def test_infer_simple_human_move():
    g = new_game()
    m = Move(((5, 0), (4, 1)))
    inf = g.infer_human_move(grid_after(g.state, m))
    assert inf.kind == "legal" and inf.move == m


def test_no_change():
    g = new_game()
    assert g.infer_human_move(g.expected_grid()).kind == "no_change"


def test_color_misdetection_is_tolerated():
    g = new_game()
    m = Move(((5, 2), (4, 3)))
    grid = grid_after(g.state, m)
    grid[0][1] = 1          # black piece classified as white
    inf = g.infer_human_move(grid)
    assert inf.kind == "legal" and inf.move == m


def test_single_noise_square_elsewhere_is_tolerated_but_not_on_move():
    g = new_game()
    m = Move(((5, 2), (4, 3)))
    grid = grid_after(g.state, m)
    grid[1][0] = 0          # a black piece far away missed by the detector
    inf = g.infer_human_move(grid)
    assert inf.kind == "legal" and inf.move == m and inf.noise_squares == [(1, 0)]
    # noise on the moved piece itself is never accepted
    grid = grid_after(g.state, m)
    grid[4][3] = 0
    assert g.infer_human_move(grid).kind == "invalid"


def test_illegal_backward_move_rejected():
    g = Game(human_first=True)
    g.start_from_grid(_grid("""
        ........
        ........
        ........
        ....w...
        ........
        ........
        ...b....
        ........"""), turn=WHITE)
    grid = _grid("""
        ........
        ........
        ........
        ........
        .....w..
        ........
        ...b....
        ........""")
    inf = g.infer_human_move(grid)
    assert inf.kind == "invalid"
    assert "not a legal move" in inf.reason


def _grid(text):
    rows = [ln.strip() for ln in text.strip().splitlines()]
    m = {".": 0, "w": 1, "b": 2}
    return [[m[ch] for ch in row] for row in rows]


def test_human_capture_and_forgotten_removal():
    g = Game(human_first=True)
    g.start_from_grid(_grid("""
        ........
        ........
        ...b....
        ....w...
        ........
        ........
        ........
        b......."""), turn=WHITE)
    # proper capture: white jumps to (1,2), black (2,3) removed by the human
    ok_grid = _grid("""
        ........
        ..w.....
        ........
        ........
        ........
        ........
        ........
        b.......""")
    inf = g.infer_human_move(ok_grid)
    assert inf.kind == "legal" and inf.move.captured == ((2, 3),)
    # human forgot to take the black piece off -> not accepted
    forgot = [row[:] for row in ok_grid]
    forgot[2][3] = 2
    inf = g.infer_human_move(forgot)
    assert inf.kind == "invalid" and inf.closest is not None
    assert inf.closest_diff == [(2, 3)]


def test_huff_flow_robot_removes_piece_and_reports_it():
    g = Game(human_first=True)
    g.start_from_grid(_grid("""
        ........
        ........
        ...b....
        ....w...
        ........
        ......w.
        ........
        .b......"""), turn=WHITE)
    # human ignores the capture (3,4)x(2,3) and plays (5,6)->(4,5)
    grid = _grid("""
        ........
        ........
        ...b....
        ....w...
        .....w..
        ........
        ........
        .b......""")
    inf = g.infer_human_move(grid)
    assert inf.kind == "huff" and inf.huff_squares == [(3, 4)]
    g.apply_human(inf)
    assert g.turn == ROBOT
    plan = g.plan_robot_turn(level=2, rng=random.Random(0))
    assert plan.huffed == (3, 4)
    assert (3, 4, 1) in plan.removed
    g.apply_robot(plan)
    assert g.state.at(3, 4) == 0


def test_robot_never_skips_capture():
    g = Game(human_first=True)
    g.start_from_grid(_grid("""
        ........
        ..b.....
        ...w....
        ........
        ........
        ........
        ......w.
        ........"""), turn=BLACK)
    plan = g.plan_robot_turn(level=3, rng=random.Random(0))
    assert plan.move.captured == ((2, 3),)
    assert plan.removed == [(2, 3, 1)]


def test_game_over_when_robot_has_no_move():
    g = Game(human_first=True)
    g.start_from_grid(_grid("""
        b.......
        .w......
        ..w.....
        ........
        ........
        ........
        ........
        ........"""), turn=BLACK)
    assert g.winner() == WHITE
    plan = g.plan_robot_turn(level=2)
    assert plan.move is None


def test_repetition_draw():
    g = Game(human_first=True)
    g.start_from_grid(_grid("""
        ........
        ........
        ........
        ........
        ........
        ........
        ........
        ........"""), turn=WHITE)
    s = GameState.from_grid(g.expected_grid(), WHITE)
    # kings shuffling back and forth
    g.state = GameState([0] * 64, WHITE)
    cells = [0] * 64
    cells[5 * 8 + 2] = 2      # white king
    cells[1 * 8 + 4] = -2     # black king
    g.state = GameState(cells, WHITE)
    g.history.clear()
    g.history[g.state.key()] += 1
    seq_w = [Move(((5, 2), (4, 3))), Move(((4, 3), (5, 2)))]
    seq_b = [Move(((1, 4), (2, 5))), Move(((2, 5), (1, 4)))]
    for i in range(4):
        g.state = g.state.apply(seq_w[i % 2]); g.history[g.state.key()] += 1
        g.state = g.state.apply(seq_b[i % 2]); g.history[g.state.key()] += 1
    assert g.winner() == "draw"
    assert s is not None


def test_full_game_human_random_via_grids():
    """Simulate the whole loop: robot plans/applies, human 'moves' by handing
    a camera grid; every inferred move must equal the move actually made."""
    rng = random.Random(7)
    g = Game(human_first=False)
    g.start_standard()
    for ply in range(120):
        if g.winner() is not None:
            break
        if g.turn == ROBOT:
            g.apply_robot(g.plan_robot_turn(level=1, time_limit=0.3, rng=rng))
        else:
            m = rng.choice(g.state.legal_moves())
            inf = g.infer_human_move(grid_after(g.state, m))
            assert inf.kind == "legal"
            assert g.state.apply(inf.move) == g.state.apply(m)
            g.apply_human(inf)


def test_unknown_squares_are_ignored_but_ambiguity_is_rejected():
    g = new_game()
    m = Move(((5, 2), (4, 3)))
    grid = grid_after(g.state, m)
    grid[0][1] = -1         # can't see a black piece far away -> fine
    grid[1][2] = 3          # occupied, colour unknown -> fine
    inf = g.infer_human_move(grid)
    assert inf.kind == "legal" and inf.move == m
    # a square the move touches must be SEEN: destination unknown -> wait
    grid = grid_after(g.state, m)
    grid[4][3] = -1
    assert not g.infer_human_move(grid).accepted
    grid[4][1] = -1
    assert not g.infer_human_move(grid).accepted


def test_unknown_square_cannot_hide_an_illegal_move():
    # review finding: man moved 2 rows straight up + unknown on (4,1) was
    # accepted as (5,0)-(4,1) with the real destination called "noise"
    g = new_game()
    grid = g.expected_grid()
    grid[5][0] = 0
    grid[3][0] = 1
    grid[4][1] = -1
    assert not g.infer_human_move(grid).accepted
    # a piece that APPEARS is never explained away as noise
    grid = g.expected_grid()
    grid[5][0] = 0
    grid[3][0] = 1
    assert not g.infer_human_move(grid).accepted


def test_captured_piece_must_be_seen_removed():
    g = Game(human_first=True)
    g.start_from_grid(_grid("""
        ........
        ........
        ........
        ........
        ...b....
        ..w.....
        ........
        b......."""), turn=WHITE)
    grid = _grid("""
        ........
        ........
        ........
        ....w...
        ...b....
        ........
        ........
        b.......""")
    grid[4][3] = -1             # camera unsure whether the black piece is gone
    assert not g.infer_human_move(grid).accepted
    grid[4][3] = 0
    inf = g.infer_human_move(grid)
    assert inf.accepted and inf.move.captured == ((4, 3),)


def test_huff_can_change_who_wins():
    # review finding: white ignores the capture, black looks blocked, but the
    # huff frees black - the robot must still get its turn
    g = Game(human_first=True)
    g.start_from_grid(_grid("""
        ........
        ........
        ........
        ....w...
        ........
        ........
        .b......
        w.w....."""), turn=WHITE)
    grid = _grid("""
        ........
        ........
        ...w....
        ........
        ........
        ........
        .b......
        w.w.....""")
    inf = g.infer_human_move(grid)
    assert inf.kind == "huff"
    g.apply_human(inf)
    assert g.pending_huff
    plan = g.plan_robot_turn(level=2, rng=random.Random(0))
    assert plan.huffed is not None and plan.move is not None


def test_force_start_crowns_men_on_far_row():
    g = Game(human_first=True)
    g.start_from_grid(_grid("""
        ...w....
        ........
        ........
        ........
        ........
        ........
        ........
        ..b....."""), turn=WHITE)
    assert g.state.at(0, 3) == 2 and g.state.at(7, 2) == -2
    assert g.winner() is None


def test_start_check_accepts_occupied_unknown_colour():
    grid = GameState.initial(1).to_grid()
    grid[0][1] = 3
    ok, _, _ = Game.check_standard_start(grid)
    assert ok
