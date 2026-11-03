import time

import numpy as np
import pytest

from chess_game.game import obs_from_labels
from chess_game.vision.reader import BoardReader

A = np.zeros(64, int)
A[:8] = 1                          # white pawns on rank 1 squares (class 1)
B = A.copy()
B[0], B[20] = 0, 1                 # one piece moved


def fake(seq):
    """classify() returning the observations of `seq` one after the other."""
    it = iter(seq)
    return lambda image: obs_from_labels(next(it), 13)


def test_needs_three_agreeing_readings():
    r = BoardReader(fake([A, A, A]), needed=3, threaded=False)
    for t in (1.0, 2.0):
        r.offer("img", t)
        assert r.result() is None
    r.offer("img", 3.0)
    assert np.argmax(r.result(), axis=1).tolist() == A.tolist()


def test_disturbance_throws_older_readings_away():
    r = BoardReader(fake([A, A, B, B, B]), needed=3, threaded=False)
    r.offer("img", 1.0)
    r.offer("img", 2.0)
    r.disturbed(2.5)               # the human moved a piece
    assert r.result() is None
    r.offer("img", 2.2)            # a frame from before the disturbance: ignored
    for t in (3.0, 4.0, 5.0):
        r.offer("img", t)
    assert np.argmax(r.result(), axis=1).tolist() == B.tolist()


def test_disagreeing_readings_wait_then_average():
    seq = [A, B] * 4
    r = BoardReader(fake(seq), needed=3, max_reads=6, threaded=False)
    for i in range(5):
        r.offer("img", float(i + 1))
        assert r.result() is None          # A/B flicker never agrees on 3
    r.offer("img", 6.0)
    assert r.result() is not None          # after max_reads: average of all


def test_same_frame_is_read_once():
    calls = []

    def classify(image):
        calls.append(image)
        return obs_from_labels(A, 13)
    r = BoardReader(classify, needed=1, threaded=False)
    r.offer("img", 1.0)
    r.offer("img", 1.0)
    assert len(calls) == 1


def test_threaded_reader_keeps_up():
    def slow(image):
        time.sleep(0.02)
        return obs_from_labels(A, 13)
    r = BoardReader(slow, needed=3, threaded=True)
    try:
        t = time.time()
        while r.result() is None and time.time() - t < 2.0:
            r.offer("img", time.time())
            time.sleep(0.03)
        assert r.result() is not None
        assert r.read_seconds == pytest.approx(0.02, abs=0.05)
    finally:
        r.stop()
