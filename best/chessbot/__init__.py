# chessbot/__init__.py

# Lazy imports to avoid loading torch if not needed
def __getattr__(name):
    if name == "ChessPolicyNet":
        from .model import ChessPolicyNet
        return ChessPolicyNet
    elif name == "ChessBot":
        from .inference import ChessBot
        return ChessBot
    elif name == "board_to_tensor":
        from .board_encoding import board_to_tensor
        return board_to_tensor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "ChessPolicyNet",
    "ChessBot",
    "board_to_tensor",
]