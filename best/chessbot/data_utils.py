import io
import json
import glob
import chess.pgn
import zstandard as zstd
from tqdm import tqdm

def stream_lichess_pgn(path):
    """Stream a .pgn.zst file efficiently"""
    dctx = zstd.ZstdDecompressor(max_window_size=2147483648)

    with open(path, 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            while True:
                try:
                    game = chess.pgn.read_game(text_stream)
                except:
                    break
                if game is None:
                    break
                yield game


def is_high_quality(game, min_elo=2000):
    """Filter out low-quality games"""
    white = game.headers.get("WhiteElo")
    black = game.headers.get("BlackElo")
    if white is None or black is None:
        return False
    return int(white) >= min_elo and int(black) >= min_elo


def build_move_index_map(pgn_dir, output_path):
    """Scan all PGNs and collect unique moves"""
    uci_set = set()

    # Find both .pgn and .zst files
    pgn_files = glob.glob(pgn_dir + "/*.zst") + glob.glob(pgn_dir + "/*.pgn")
    print(f"Scanning {len(pgn_files)} PGN files for unique moves...")

    for fp in pgn_files:
        if fp.endswith('.zst'):
            # Compressed file
            for game in stream_lichess_pgn(fp):
                if not is_high_quality(game):
                    continue
                for move in game.mainline_moves():
                    uci_set.add(move.uci())
        else:
            # Regular PGN file
            import chess.pgn
            with open(fp, 'r') as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    if not is_high_quality(game):
                        continue
                    for move in game.mainline_moves():
                        uci_set.add(move.uci())

    uci_list = sorted(list(uci_set))
    move_index_map = {uci: i for i, uci in enumerate(uci_list)}

    with open(output_path, "w") as f:
        json.dump(move_index_map, f)

    print(f"Saved move map with {len(move_index_map)} moves.")
    return move_index_map