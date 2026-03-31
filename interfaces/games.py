import numpy as np

def create_board():
    return np.zeros((3,3), dtype=int)  # 0 = vide, 1 = X, 2 = O

def display_board(board):
    chars = {0: " ", 1: "X", 2: "O"}
    print("\n".join(["|".join([chars[cell] for cell in row]) for row in board]))
    print()


def check_winner(board):
    # lignes et colonnes
    for i in range(3):
        if len(set(board[i,:])) == 1 and board[i,0] != 0:
            return board[i,0]
        if len(set(board[:,i])) == 1 and board[0,i] != 0:
            return board[0,i]
    # diagonales
    if board[0,0] == board[1,1] == board[2,2] != 0:
        return board[0,0]
    if board[0,2] == board[1,1] == board[2,0] != 0:
        return board[0,2]
    # match nul
    if 0 not in board:
        return 0  # 0 = match nul
    return None  # partie non terminée

def human_move(board, player):
    while True:
        move = input(f"Joueur {player} (ligne,colonne) : ")
        try:
            i,j = map(int, move.split(","))
            if board[i,j] == 0:
                board[i,j] = player
                break
            else:
                print("Case déjà occupée.")
        except:
            print("Mouvement invalide, format : ligne,colonne (0-2)")

def ml_ai_move(board, player, model_x, model_draw):
    best_score = -1
    best_move = None
    for i in range(3):
        for j in range(3):
            if board[i,j] == 0:
                board[i,j] = player
                x = np.array([board.flatten()]).reshape(1, -1)  # transformer en 2D
                score = model_x.predict_proba(x)[0][1]  # probabilité de victoire
                draw_score = model_draw.predict_proba(x)[0][1]  # probabilité de match nul
                total_score = score + 0.5*draw_score  # pondération
                if total_score > best_score:
                    best_score = total_score
                    best_move = (i,j)
                board[i,j] = 0  # reset
    board[best_move] = player

def evaluate_board(board, model_x, model_draw, player):
    x = np.array([board.flatten()]).reshape(1, -1)
    win_prob = model_x.predict_proba(x)[0][1] if player == 1 else 1 - model_x.predict_proba(x)[0][1]
    draw_prob = model_draw.predict_proba(x)[0][1]
    return win_prob + 0.5*draw_prob

def minimax(board, depth, alpha, beta, maximizingPlayer, model_x, model_draw, player):
    winner = check_winner(board)
    if depth == 0 or winner is not None:
        if winner == player:
            return 1
        elif winner == 0:  # match nul
            return 0
        elif winner is not None:
            return -1
        else:
            return evaluate_board(board, model_x, model_draw, player)
    
    if maximizingPlayer:
        max_eval = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i,j] == 0:
                    board[i,j] = player
                    eval = minimax(board, depth-1, alpha, beta, False, model_x, model_draw, player)
                    board[i,j] = 0
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = float('inf')
        opponent = 2 if player == 1 else 1
        for i in range(3):
            for j in range(3):
                if board[i,j] == 0:
                    board[i,j] = opponent
                    eval = minimax(board, depth-1, alpha, beta, True, model_x, model_draw, player)
                    board[i,j] = 0
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval

def hybrid_ai_move(board, player, model_x, model_draw):
    best_score = -float('inf')
    best_move = None
    for i in range(3):
        for j in range(3):
            if board[i,j] == 0:
                board[i,j] = player
                score = minimax(board, 3, -float('inf'), float('inf'), False, model_x, model_draw, player)
                board[i,j] = 0
                if score > best_score:
                    best_score = score
                    best_move = (i,j)
    board[best_move] = player

def play_game(mode, model_x=None, model_draw=None):
    board = create_board()
    current_player = 1
    while True:
        display_board(board)
        winner = check_winner(board)
        if winner is not None:
            if winner == 0:
                print("Match nul !")
            else:
                print(f"Le joueur {winner} a gagné !")
            break
        
        if mode == "human":
            human_move(board, current_player)
        elif mode == "ml":
            ml_ai_move(board, current_player, model_x, model_draw)
        elif mode == "hybrid":
            hybrid_ai_move(board, current_player, model_x, model_draw)
        
        current_player = 2 if current_player == 1 else 1
        
if __name__ == "__main__":
    play_game()