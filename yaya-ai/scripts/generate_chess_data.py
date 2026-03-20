"""
generate_chess_data.py
----------------------
Generates 100 chess games using weighted random play (prefers captures, checks,
center moves) and saves them in two formats:

  data/chess/games.pgn       — raw PGN for pretraining
  data/chess/chess_sft.jsonl — instruction fine-tuning examples

Requires: pip install chess
"""

import json
import os
import random
import sys
import datetime

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    import chess
    import chess.pgn
except ImportError:
    print("ERROR: python-chess is not installed.")
    print("Install it with:  pip install chess")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.join(SCRIPT_DIR, "..")
CHESS_DIR   = os.path.join(REPO_ROOT, "data", "chess")
PGN_PATH    = os.path.join(CHESS_DIR, "games.pgn")
SFT_PATH    = os.path.join(CHESS_DIR, "chess_sft.jsonl")

os.makedirs(CHESS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_GAMES        = 100
SYSTEM_PROMPT    = (
    "You are Yaya, a chess-playing AI. "
    "You understand chess deeply and can explain moves clearly."
)

# Center squares (e4, d4, e5, d5 and surrounding area)
CENTER_SQUARES = {
    chess.E4, chess.D4, chess.E5, chess.D5,
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4, chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}

# ---------------------------------------------------------------------------
# Opening recognition
# ---------------------------------------------------------------------------
OPENING_TABLE: list[tuple[tuple[str, ...], str]] = [
    # King's Pawn openings
    (("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"),           "Ruy López (Spanish Game)"),
    (("e2e4", "e7e5", "g1f3", "b8c6", "f1c4"),           "Italian Game"),
    (("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"),   "Two Knights Defense"),
    (("e2e4", "e7e5", "g1f3", "b8c6", "d2d4"),            "Scotch Game"),
    (("e2e4", "e7e5", "f2f4"),                             "King's Gambit"),
    (("e2e4", "e7e5", "g1f3", "g8f6"),                    "Petrov's Defense"),
    (("e2e4", "c7c5"),                                     "Sicilian Defense"),
    (("e2e4", "c7c5", "g1f3", "d7d6", "d2d4"),            "Sicilian Defense — Open"),
    (("e2e4", "c7c5", "g1f3", "b8c6"),                    "Sicilian Defense — Classical"),
    (("e2e4", "c7c5", "g1f3", "e7e6"),                    "Sicilian Defense — Kan/Taimanov"),
    (("e2e4", "e7e6"),                                     "French Defense"),
    (("e2e4", "e7e6", "d2d4", "d7d5", "b1c3"),            "French Defense — Classical"),
    (("e2e4", "e7e6", "d2d4", "d7d5", "e4e5"),            "French Defense — Advance"),
    (("e2e4", "c7c6"),                                     "Caro-Kann Defense"),
    (("e2e4", "d7d5"),                                     "Scandinavian Defense"),
    (("e2e4", "g8f6"),                                     "Alekhine's Defense"),
    # Queen's Pawn openings
    (("d2d4", "d7d5"),                                     "Queen's Pawn Game"),
    (("d2d4", "d7d5", "c2c4"),                             "Queen's Gambit"),
    (("d2d4", "d7d5", "c2c4", "e7e6"),                    "Queen's Gambit Declined"),
    (("d2d4", "d7d5", "c2c4", "c7c6"),                    "Slav Defense"),
    (("d2d4", "d7d5", "c2c4", "d5c4"),                    "Queen's Gambit Accepted"),
    (("d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"),   "Nimzo-Indian Defense"),
    (("d2d4", "g8f6", "c2c4", "g7g6"),                    "King's Indian Defense"),
    (("d2d4", "g8f6", "c2c4", "e7e6"),                    "Queen's Indian Defense"),
    (("d2d4", "g8f6", "g1f3", "g7g6"),                    "King's Indian Attack"),
    (("d2d4", "d7d5", "g1f3", "g8f6", "c2c4"),            "Queen's Gambit — Symmetrical"),
    # Flank openings
    (("g1f3",),                                            "Réti Opening"),
    (("c2c4",),                                            "English Opening"),
    (("g2g3",),                                            "King's Fianchetto Opening"),
    (("b2b3",),                                            "Larsen's Opening"),
    (("f2f4",),                                            "Bird's Opening"),
]


def detect_opening(board_moves: list[str]) -> str:
    """Return the name of the opening from the move sequence, or 'Unknown Opening'."""
    best_match = "Unknown Opening"
    best_len   = 0
    for pattern, name in OPENING_TABLE:
        plen = len(pattern)
        if (plen <= len(board_moves)
                and tuple(board_moves[:plen]) == pattern
                and plen > best_len):
            best_match = name
            best_len   = plen
    return best_match


# ---------------------------------------------------------------------------
# Move-quality heuristic weights
# ---------------------------------------------------------------------------
def score_move(board: chess.Board, move: chess.Move) -> float:
    """
    Assign a weight to a legal move so that weighted random selection
    produces plausible chess play.

    Priorities (cumulative score):
      1. Captures        — prefer capturing higher-value pieces
      2. Checks          — modest bonus for giving check
      3. Center control  — bonus for moving to / occupying central squares
      4. Promotion       — big bonus for pawn promotions
      5. Castling        — small bonus for king safety
    """
    PIECE_VALUE = {
        chess.PAWN:   1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK:   5,
        chess.QUEEN:  9,
        chess.KING:   0,
    }

    score = 1.0  # base weight (every move has some probability)

    # Capture bonus
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        if victim is not None:
            score += PIECE_VALUE.get(victim.piece_type, 0) * 2.0
        else:
            # en-passant
            score += 2.0

    # Push the move onto a scratch board to test for check / promotion
    board.push(move)
    if board.is_check():
        score += 2.5
    board.pop()

    # Center control
    if move.to_square in CENTER_SQUARES:
        score += 1.5

    # Promotion
    if move.promotion is not None:
        score += 20.0 if move.promotion == chess.QUEEN else 5.0

    # Castling
    if board.is_castling(move):
        score += 3.0

    return score


def weighted_random_move(board: chess.Board) -> chess.Move:
    """Pick a legal move with probability proportional to its heuristic score."""
    legal = list(board.legal_moves)
    weights = [score_move(board, m) for m in legal]
    return random.choices(legal, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Chess explanation helpers
# ---------------------------------------------------------------------------
def piece_name(piece: chess.Piece | None) -> str:
    if piece is None:
        return "piece"
    names = {
        chess.PAWN:   "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK:   "rook",
        chess.QUEEN:  "queen",
        chess.KING:   "king",
    }
    return names.get(piece.piece_type, "piece")


def color_name(turn: bool) -> str:
    return "White" if turn else "Black"


def explain_move(board: chess.Board, move: chess.Move) -> str:
    """
    Generate a plain-language explanation of why a move is played.
    board must be BEFORE the move is pushed.
    """
    reasons: list[str] = []
    mover_color = color_name(board.turn)
    mover       = board.piece_at(move.from_square)
    mover_name  = piece_name(mover)
    to_name     = chess.square_name(move.to_square)
    from_name   = chess.square_name(move.from_square)

    # Capture?
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        if victim:
            reasons.append(
                f"capturing the {piece_name(victim)} on {to_name}"
            )
        else:
            reasons.append(f"capturing en passant on {to_name}")

    # Castling?
    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) > 4 else "queenside"
        reasons.append(f"castling {side} to improve king safety")

    # Check?
    board.push(move)
    if board.is_check():
        reasons.append("giving check")
    if board.is_checkmate():
        reasons.append("delivering checkmate")
    board.pop()

    # Center control?
    if move.to_square in CENTER_SQUARES and not reasons:
        reasons.append(f"controlling the center from {to_name}")

    # Promotion?
    if move.promotion == chess.QUEEN:
        reasons.append(f"promoting the pawn to a queen on {to_name}")

    # Development (first few moves, piece moves from back rank)?
    if mover and mover.piece_type in (chess.KNIGHT, chess.BISHOP):
        if chess.square_rank(move.from_square) in (0, 7):
            reasons.append(f"developing the {mover_name} to {to_name}")

    if not reasons:
        reasons.append(f"moving the {mover_name} from {from_name} to {to_name}")

    return "This move is good because it " + ", and ".join(reasons) + "."


def result_to_text(result: str) -> str:
    mapping = {"1-0": "White wins", "0-1": "Black wins", "1/2-1/2": "Draw", "*": "Unfinished"}
    return mapping.get(result, result)


# ---------------------------------------------------------------------------
# Game generator
# ---------------------------------------------------------------------------
def play_game(game_id: int) -> tuple[chess.pgn.Game, list[dict]]:
    """
    Play one game and return (pgn_game, list_of_sft_examples).
    Each game contributes:
      - One SFT example per move (move prediction)
      - One game-level commentary example
    """
    board          = chess.Board()
    pgn_game       = chess.pgn.Game()
    node           = pgn_game
    move_uci_list: list[str] = []
    sft_examples:  list[dict] = []

    # Random opening variety: sometimes start from a known ECO variation
    # by seeding the first few moves with a randomly chosen pattern.
    if random.random() < 0.70 and OPENING_TABLE:
        pattern, _ = random.choice(OPENING_TABLE)
        seed_moves: list[chess.Move] = []
        for uci in pattern:
            try:
                m = chess.Move.from_uci(uci)
                if m in board.legal_moves:
                    seed_moves.append(m)
                else:
                    break
            except ValueError:
                break
        # Apply seed moves
        for m in seed_moves:
            move_uci_list.append(m.uci())
            node = node.add_variation(m)
            board.push(m)

    max_halfmoves = random.randint(40, 120)  # varied game lengths

    while not board.is_game_over() and board.fullmove_number * 2 <= max_halfmoves:
        fen        = board.fen()
        turn_color = color_name(board.turn)
        move       = weighted_random_move(board)
        san        = board.san(move)
        explanation = explain_move(board, move)

        # --- Move prediction SFT example ---
        sft_examples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": (
                    f"Chess position (FEN): {fen}. "
                    f"It is {turn_color}'s turn. What is the best move?"
                )},
                {"role": "assistant", "content": (
                    f"I play {san}. {explanation}"
                )},
            ]
        })

        move_uci_list.append(move.uci())
        node = node.add_variation(move)
        board.push(move)

    # Final game result
    result = board.result()
    pgn_game.headers["Event"]  = f"Yaya Training Game {game_id + 1}"
    pgn_game.headers["White"]  = "YayaEngine"
    pgn_game.headers["Black"]  = "YayaEngine"
    pgn_game.headers["Result"] = result
    pgn_game.headers["Date"]   = datetime.date.today().strftime("%Y.%m.%d")

    opening_name = detect_opening(move_uci_list)
    pgn_game.headers["Opening"] = opening_name

    # --- Game commentary SFT example ---
    pgn_str = str(pgn_game)
    result_text = result_to_text(result)
    num_moves   = board.fullmove_number - 1
    comment     = (
        f"This game opened with the {opening_name}. "
        f"After {num_moves} moves, the game ended with {result_text}. "
    )
    if result == "1-0":
        comment += (
            "White maintained pressure throughout and converted their advantage. "
            "Key themes included piece activity and king safety."
        )
    elif result == "0-1":
        comment += (
            "Black defended resourcefully and found a winning plan. "
            "The endgame technique was decisive."
        )
    elif result == "1/2-1/2":
        comment += (
            "Both sides played accurately and the game ended in a balanced draw. "
            "Neither player could create a decisive imbalance."
        )
    else:
        comment += "The game was agreed drawn or abandoned."

    sft_examples.append({
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": (
                f"Here is a chess game in PGN format:\n\n{pgn_str}\n\n"
                "Please provide commentary on this game."
            )},
            {"role": "assistant", "content": comment},
        ]
    })

    # --- Tactics example: last position before game-ending move (if checkmate) ---
    if board.is_checkmate() and len(move_uci_list) >= 2:
        # Replay to one move before the final move
        pre_board = chess.Board()
        for uci in move_uci_list[:-1]:
            pre_board.push(chess.Move.from_uci(uci))
        final_move = chess.Move.from_uci(move_uci_list[-1])
        if final_move in pre_board.legal_moves:
            fen_tactic    = pre_board.fen()
            san_tactic    = pre_board.san(final_move)
            tactic_color  = color_name(pre_board.turn)
            sft_examples.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": (
                        f"Chess position (FEN): {fen_tactic}. "
                        f"It is {tactic_color}'s turn. "
                        "There is a decisive tactic here. What is the best move and why?"
                    )},
                    {"role": "assistant", "content": (
                        f"The winning move is {san_tactic}. "
                        f"This delivers checkmate in one move, ending the game immediately. "
                        f"{tactic_color} should play {san_tactic} to win."
                    )},
                ]
            })

    return pgn_game, sft_examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    random.seed(42)

    print(f"Generating {NUM_GAMES} chess games...")
    print(f"  PGN  -> {os.path.abspath(PGN_PATH)}")
    print(f"  SFT  -> {os.path.abspath(SFT_PATH)}")
    print()

    all_sft: list[dict] = []

    with open(PGN_PATH, "w", encoding="utf-8") as pgn_file:
        for i in range(NUM_GAMES):
            pgn_game, sft_examples = play_game(i)
            all_sft.extend(sft_examples)

            # Write PGN (two games are separated by a blank line)
            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
            pgn_file.write(pgn_game.accept(exporter))
            pgn_file.write("\n\n")

            opening = pgn_game.headers.get("Opening", "Unknown Opening")
            result  = pgn_game.headers.get("Result", "*")
            print(
                f"  [{i+1:3d}/{NUM_GAMES}] {opening:<40s}  "
                f"result={result}  sft_so_far={len(all_sft)}"
            )

    # Write SFT JSONL
    with open(SFT_PATH, "w", encoding="utf-8") as sft_file:
        for ex in all_sft:
            sft_file.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print()
    print(f"Done.")
    print(f"  Games written  : {NUM_GAMES}")
    print(f"  SFT examples   : {len(all_sft)}")
    print(f"  PGN file       : {os.path.abspath(PGN_PATH)}")
    print(f"  SFT JSONL file : {os.path.abspath(SFT_PATH)}")


if __name__ == "__main__":
    main()
