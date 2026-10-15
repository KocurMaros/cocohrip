import os

WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# rgb (pygame drawing colours - NOT the rules-engine colours, see
# checkers_game.checkers.rules.WHITE / BLACK)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (40, 200, 60)
ORANGE = (255, 150, 0)
GREY = (220, 217, 209)
BROWN = (124, 84, 5)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts):
    """Find a file shipped with the package.

    Works both when installed (share/ or site-packages) and when the node is
    started from ~/<ws> with the source tree in src/ (the old hard-coded
    'src/checkers_game/checkers_game/...' relative path)."""
    candidates = [
        os.path.join(_PKG_DIR, *parts),
        os.path.join(os.getcwd(), "src", "checkers_game", "checkers_game", *parts),
        os.path.join(os.getcwd(), "src", "cocohrip", "checkers_game", "checkers_game", *parts),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


_CROWN = None


def get_crown():
    """Crown image (lazy, so importing constants does not need pygame)."""
    global _CROWN
    if _CROWN is None:
        import pygame
        path = resource_path("assets", "crown.png")
        try:
            _CROWN = pygame.transform.scale(pygame.image.load(path), (44, 25))
        except Exception:  # noqa: BLE001 - missing asset must not kill the game
            _CROWN = False
    return _CROWN or None
