import numpy as np
import chess


def board_to_matrix(board: chess.Board):
    # 8x8 is a size of the chess board.
    # 12 = number of unique pieces.
    # 13th board for the active player (1 for white, -1 for black)
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the 13th board with the active player
    player_value = 1 if board.turn else -1
    matrix[12, :, :] = player_value  # Fill the entire 8x8 board with the player's value

    return matrix


def create_input_and_value(games, normalize_by=9.0):
    X, y_policy, y_value = [], [], []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            # encoding policy
            y_policy.append(move.uci())
            # encoding value
            before = material_count(board)
            board.push(move)
            after = material_count(board)
            # différence normalisée
            y_value.append((after - before) / normalize_by)
            # input
            X.append(board_to_matrix(board))
            
    return (
        np.array(X, dtype=np.float32),
        np.array(y_policy, dtype=object),  # strings pour l'instant
        np.array(y_value, dtype=np.float32)
    )

_piece_values = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0
}

def material_count(board: chess.Board) -> int:
    total = 0
    for sq, piece in board.piece_map().items():
        total += _piece_values[piece.piece_type] * (1 if piece.color == board.turn else -1)
    return total
    
def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.int64), move_to_int


