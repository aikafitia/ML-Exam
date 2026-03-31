import itertools
import pandas as pd

# Vérifier gagnant

def check_winner(board):
    wins = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    
    for a, b, c in wins:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    
    return 0

# Minimax

def minimax(board, is_x_turn):
    winner = check_winner(board)

    if winner == 1:
        return 1   # X gagne
    elif winner == -1:
        return -1  # O gagne
    elif 0 not in board:
        return 0   # nul

    if is_x_turn:
        best = -999
        for i in range(9):
            if board[i] == 0:
                board[i] = 1
                score = minimax(board, False)
                board[i] = 0
                best = max(best, score)
        return best
    else:
        best = 999
        for i in range(9):
            if board[i] == 0:
                board[i] = -1
                score = minimax(board, True)
                board[i] = 0
                best = min(best, score)
        return best


# Vérifier état valide

def is_valid(board):
    return abs(board.count(1) - board.count(-1)) <= 1


# Encodage features (18 colonnes)

def encode(board):
    features = []
    for cell in board:
        features.append(1 if cell == 1 else 0)  # X
        features.append(1 if cell == -1 else 0) # O
    return features


# Génération dataset

def generate_dataset():
    data = []

    print("Génération du dataset...")

    for board in itertools.product([0, 1, -1], repeat=9):
        board = list(board)

        # Etat valide
        if not is_valid(board):
            continue

        # Tour de X uniquement
        if board.count(1) != board.count(-1):
            continue

        result = minimax(board, True)

        x_wins = 1 if result == 1 else 0
        is_draw = 1 if result == 0 else 0

        row = encode(board) + [x_wins, is_draw]
        data.append(row)

    columns = []
    for i in range(9):
        columns.append(f"c{i}_x")
        columns.append(f"c{i}_o")

    columns += ["x_wins", "is_draw"]

    df = pd.DataFrame(data, columns=columns)

    df.to_csv("../ressources/dataset.csv", sep=";", index=False)

    print("Dataset généré avec succès !")
    print(f"Nombre de lignes : {len(df)}")


if __name__ == "__main__":
    generate_dataset()