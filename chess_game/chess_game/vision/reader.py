"""Reading the board in the background while it is still.

The game loop hands every new camera frame to the reader (offer) and tells
it when the board was disturbed (motion, a hand).  A worker thread
classifies the newest frame whenever it is free; readings of frames taken
before the last disturbance are thrown away.  result() gives the average of
the last `needed` readings once they agree on empty / white / black on every
square - so the reading is ready as soon as the board has been still long
enough, the game window never waits for the classifier, and one odd frame
(glare, a shadow passing) cannot decide a move.
"""

import threading
import time

import numpy as np

from chess_game.game import collapse_to_colour


class BoardReader:
    def __init__(self, classify, needed=3, max_reads=8, threaded=True):
        """classify(image) -> (64, C) probabilities."""
        self.classify = classify
        self.needed = max(1, int(needed))
        self.max_reads = max(self.needed, int(max_reads))
        self.threaded = threaded
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._frame = None                  # (image, time) waiting to be read
        self._last_read_t = -1.0
        self._reads = []                    # [(frame time, probabilities)]
        self._disturbed_at = 0.0
        self.busy = False                   # classifying right now
        self.read_seconds = 0.0             # duration of the last classification
        self._running = threaded
        if threaded:
            threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------ input
    def offer(self, image, t):
        """The newest frame (taken at time t)."""
        with self._lock:
            self._frame = (image, t)
        if self.threaded:
            self._wake.set()
        else:
            self._step()

    def disturbed(self, t):
        """Motion or a hand at time t: readings of older frames are void."""
        with self._lock:
            if t > self._disturbed_at:
                self._disturbed_at = t
                self._reads = [r for r in self._reads if r[0] >= t]

    # ------------------------------------------------------------ output
    def result(self, since=None):
        """Average of the last `needed` readings of frames taken after
        `since` (default: the last disturbance) if they agree on empty /
        white / black everywhere; after max_reads disagreeing readings the
        average of all of them.  None = not ready yet."""
        with self._lock:
            since = self._disturbed_at if since is None else max(since, self._disturbed_at)
            reads = [p for t, p in self._reads if t >= since]
        if len(reads) < self.needed:
            return None
        last = reads[-self.needed:]
        colours = [np.argmax(_colours(p), axis=1) for p in last]
        if all(np.array_equal(colours[0], c) for c in colours[1:]):
            return np.mean(last, axis=0)
        if len(reads) >= self.max_reads:        # keeps flickering: decide on everything
            return np.mean(reads, axis=0)
        return None

    def reading(self):
        """True while frames of the current still period are being read."""
        with self._lock:
            pending = self._frame is not None and self._frame[1] > self._last_read_t
        return self.busy or pending

    def clear(self):
        with self._lock:
            self._reads = []

    def stop(self):
        self._running = False
        self._wake.set()

    # ------------------------------------------------------------ worker
    def _loop(self):
        while self._running:
            self._wake.wait(0.5)
            self._wake.clear()
            while self._running and self._step():
                pass

    def _step(self):
        """Read the newest frame if it is new and after the last
        disturbance.  Returns True if a frame was read."""
        with self._lock:
            if self._frame is None:
                return False
            image, t = self._frame
            if t <= self._last_read_t or t < self._disturbed_at:
                return False
            self._last_read_t = t
            self.busy = True
        t0 = time.perf_counter()
        try:
            probs = self.classify(image)
        finally:
            self.read_seconds = time.perf_counter() - t0
            self.busy = False
        with self._lock:
            if t >= self._disturbed_at:
                self._reads.append((t, probs))
                self._reads = self._reads[-self.max_reads:]
        return True


def _colours(p):
    return p if p.shape[1] == 3 else collapse_to_colour(p)
