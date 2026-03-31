import numpy as np
import pickle
import os
from typing import Tuple, Optional, List

class TicTacToe:
    """Jeu de Morpion avec interface pour différents modes de jeu"""
    
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None
        
    def display_board(self):
        """Affiche le plateau de jeu"""
        print('\n')
        for i in range(0, 9, 3):
            print(' | '.join(self.board[i:i+3]))
            if i < 6:
                print('---------')
        print('\n')
    
    def make_move(self, position: int, player: str) -> bool:
        """Effectue un mouvement si la case est libre"""
        if self.board[position] == ' ':
            self.board[position] = player
            if self.check_winner(position, player):
                self.current_winner = player
            return True
        return False
    
    def check_winner(self, position: int, player: str) -> bool:
        """Vérifie si le joueur a gagné"""
        # Vérifier la ligne
        row = position // 3
        if all(self.board[row*3 + i] == player for i in range(3)):
            return True
        
        # Vérifier la colonne
        col = position % 3
        if all(self.board[col + 3*i] == player for i in range(3)):
            return True
        
        # Vérifier les diagonales
        if position % 2 == 0:
            if all(self.board[i] == player for i in [0, 4, 8]):
                return True
            if all(self.board[i] == player for i in [2, 4, 6]):
                return True
        
        return False
    
    def get_available_moves(self) -> List[int]:
        """Retourne la liste des cases libres"""
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def is_draw(self) -> bool:
        """Vérifie si le plateau est plein sans gagnant"""
        return ' ' not in self.board and self.current_winner is None
    
    def reset(self):
        """Réinitialise le jeu"""
        self.board = [' ' for _ in range(9)]
        self.current_winner = None
    
    def board_to_features(self) -> np.ndarray:
        """Convertit le plateau en features pour le modèle ML"""
        # Encodage: X -> 1, O -> -1, vide -> 0
        features = []
        for cell in self.board:
            if cell == 'X':
                features.append(1)
            elif cell == 'O':
                features.append(-1)
            else:
                features.append(0)
        return np.array(features).reshape(1, -1)


class MLModel:
    """Wrapper pour les modèles ML"""
    
    def __init__(self, model_path: str = None):
        self.x_wins_model = None
        self.is_draw_model = None
        self.load_models(model_path)
    
    def load_models(self, model_path: str = None):
        """Charge les modèles entraînés"""
        # Chercher les modèles dans le dossier notebook
        if model_path is None:
            model_path = os.path.join('notebook', 'models')
        
        try:
            # Essayer de charger les modèles depuis les fichiers .pkl
            if os.path.exists(os.path.join(model_path, 'x_wins_model.pkl')):
                with open(os.path.join(model_path, 'x_wins_model.pkl'), 'rb') as f:
                    self.x_wins_model = pickle.load(f)
            
            if os.path.exists(os.path.join(model_path, 'is_draw_model.pkl')):
                with open(os.path.join(model_path, 'is_draw_model.pkl'), 'rb') as f:
                    self.is_draw_model = pickle.load(f)
            else:
                print("Modèles ML non trouvés, utilisation d'une heuristique simple")
        except Exception as e:
            print(f"Erreur chargement modèles: {e}")
            print("Utilisation d'une heuristique simple")
    
    def evaluate_position(self, board_features: np.ndarray) -> Tuple[float, float]:
        """Évalue une position avec les modèles ML"""
        if self.x_wins_model is not None and self.is_draw_model is not None:
            try:
                x_wins_prob = self.x_wins_model.predict_proba(board_features)[0][1]
                is_draw_prob = self.is_draw_model.predict_proba(board_features)[0][1]
                return x_wins_prob, is_draw_prob
            except:
                pass
        
        # Heuristique simple si modèles non disponibles
        return self._heuristic_evaluation(board_features[0])
    
    def _heuristic_evaluation(self, board) -> float:
        """Évaluation heuristique simple"""
        # Compter les alignements potentiels
        lines = [
            [0,1,2], [3,4,5], [6,7,8],  # lignes
            [0,3,6], [1,4,7], [2,5,8],  # colonnes
            [0,4,8], [2,4,6]            # diagonales
        ]
        
        x_score = 0
        o_score = 0
        
        for line in lines:
            x_count = sum(1 for i in line if board[i] == 1)
            o_count = sum(1 for i in line if board[i] == -1)
            
            if x_count == 2 and o_count == 0:
                x_score += 10
            elif x_count == 1 and o_count == 0:
                x_score += 1
            elif o_count == 2 and x_count == 0:
                o_score += 10
            elif o_count == 1 and x_count == 0:
                o_score += 1
        
        return x_score - o_score


class MinimaxAlphaBeta:
    """Implémentation de Minimax avec élagage alpha-beta"""
    
    def __init__(self, ml_model: MLModel, depth_limit: int = 3):
        self.ml_model = ml_model
        self.depth_limit = depth_limit
    
    def get_best_move(self, game: TicTacToe, player: str) -> int:
        """Trouve le meilleur coup avec minimax + alpha-beta"""
        best_score = -float('inf')
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        
        for move in game.get_available_moves():
            game.make_move(move, player)
            
            if player == 'X':
                score = self._minimax(game, 0, alpha, beta, False, 'X')
            else:
                score = self._minimax(game, 0, alpha, beta, False, 'X')
            
            game.board[move] = ' '
            game.current_winner = None
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
        
        return best_move
    
    def _minimax(self, game: TicTacToe, depth: int, alpha: float, beta: float, 
                 is_maximizing: bool, ai_player: str) -> float:
        """Algorithme Minimax avec élagage alpha-beta"""
        
        # Vérifier les conditions terminales
        if game.current_winner == ai_player:
            return 100 - depth
        elif game.current_winner is not None:  # L'adversaire a gagné
            return -100 + depth
        elif game.is_draw():
            return 0
        
        # Limite de profondeur atteinte : utiliser l'IA ML
        if depth >= self.depth_limit:
            features = game.board_to_features()
            x_wins, is_draw = self.ml_model.evaluate_position(features)
            # Convertir les probabilités en score
            if ai_player == 'X':
                return x_wins * 100
            else:
                return (1 - x_wins) * 100
        
        if is_maximizing:
            best_score = -float('inf')
            for move in game.get_available_moves():
                game.make_move(move, ai_player)
                score = self._minimax(game, depth + 1, alpha, beta, False, ai_player)
                game.board[move] = ' '
                game.current_winner = None
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            opponent = 'O' if ai_player == 'X' else 'X'
            for move in game.get_available_moves():
                game.make_move(move, opponent)
                score = self._minimax(game, depth + 1, alpha, beta, True, ai_player)
                game.board[move] = ' '
                game.current_winner = None
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score


class RandomAI:
    """IA qui joue aléatoirement"""
    
    def get_move(self, game: TicTacToe) -> int:
        import random
        moves = game.get_available_moves()
        return random.choice(moves) if moves else None


class MLBasedAI:
    """IA basée sur les modèles ML pour évaluer chaque position"""
    
    def __init__(self, ml_model: MLModel):
        self.ml_model = ml_model
    
    def get_move(self, game: TicTacToe, player: str) -> int:
        """Choisit le meilleur coup en fonction de l'évaluation ML"""
        best_score = -float('inf')
        best_move = None
        
        for move in game.get_available_moves():
            # Simuler le coup
            game.make_move(move, player)
            features = game.board_to_features()
            x_wins, is_draw = self.ml_model.evaluate_position(features)
            game.board[move] = ' '
            game.current_winner = None
            
            # Score en fonction du joueur
            if player == 'X':
                score = x_wins
            else:
                score = 1 - x_wins
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move


class GameInterface:
    """Interface principale du jeu"""
    
    def __init__(self):
        self.game = TicTacToe()
        self.ml_model = MLModel()
        self.hybrid_ai = MinimaxAlphaBeta(self.ml_model, depth_limit=3)
        self.ml_ai = MLBasedAI(self.ml_model)
        self.random_ai = RandomAI()
    
    def play_vs_human(self):
        """Mode: Humain vs Humain"""
        print("\n=== Mode: Humain vs Humain ===")
        current_player = 'X'
        
        while not self.game.current_winner and not self.game.is_draw():
            self.game.display_board()
            print(f"Joueur {current_player}, à vous de jouer")
            
            try:
                move = int(input("Choisissez une case (0-8): "))
                if move not in self.game.get_available_moves():
                    print("Case invalide, réessayez")
                    continue
            except ValueError:
                print("Veuillez entrer un nombre entre 0 et 8")
                continue
            
            self.game.make_move(move, current_player)
            current_player = 'O' if current_player == 'X' else 'X'
        
        self._display_result()
    
    def play_vs_ml_ai(self):
        """Mode: Humain vs IA ML"""
        print("\n=== Mode: Humain vs IA ML ===")
        print("Vous jouez X, l'IA joue O")
        current_player = 'X'
        
        while not self.game.current_winner and not self.game.is_draw():
            self.game.display_board()
            
            if current_player == 'X':  # Tour de l'humain
                try:
                    move = int(input("Votre coup (0-8): "))
                    if move not in self.game.get_available_moves():
                        print("Case invalide, réessayez")
                        continue
                except ValueError:
                    print("Veuillez entrer un nombre entre 0 et 8")
                    continue
            else:  # Tour de l'IA ML
                print("L'IA ML réfléchit...")
                move = self.ml_ai.get_move(self.game, 'O')
                print(f"L'IA joue en case {move}")
            
            self.game.make_move(move, current_player)
            current_player = 'O' if current_player == 'X' else 'X'
        
        self._display_result()
    
    def play_vs_hybrid_ai(self):
        """Mode: Humain vs IA Hybride (Minimax+ML)"""
        print("\n=== Mode: Humain vs IA Hybride (Minimax + ML) ===")
        print("Vous jouez X, l'IA hybride joue O")
        current_player = 'X'
        
        while not self.game.current_winner and not self.game.is_draw():
            self.game.display_board()
            
            if current_player == 'X':  # Tour de l'humain
                try:
                    move = int(input("Votre coup (0-8): "))
                    if move not in self.game.get_available_moves():
                        print("Case invalide, réessayez")
                        continue
                except ValueError:
                    print("Veuillez entrer un nombre entre 0 et 8")
                    continue
            else:  # Tour de l'IA Hybride
                print("L'IA hybride réfléchit (Minimax profondeur 3)...")
                move = self.hybrid_ai.get_best_move(self.game, 'O')
                print(f"L'IA joue en case {move}")
            
            self.game.make_move(move, current_player)
            current_player = 'O' if current_player == 'X' else 'X'
        
        self._display_result()
    
    def _display_result(self):
        """Affiche le résultat final"""
        self.game.display_board()
        if self.game.current_winner:
            print(f"🏆 Victoire du joueur {self.game.current_winner} !")
        else:
            print("🤝 Match nul !")
    
    def main_menu(self):
        """Menu principal"""
        while True:
            print("\n" + "="*40)
            print("          JEU DU MORPION")
            print("="*40)
            print("1. Humain vs Humain")
            print("2. Humain vs IA (ML)")
            print("3. Humain vs IA (Hybride Minimax+ML)")
            print("4. Quitter")
            print("="*40)
            
            choice = input("\nChoisissez une option (1-4): ")
            
            if choice == '1':
                self.game.reset()
                self.play_vs_human()
            elif choice == '2':
                self.game.reset()
                self.play_vs_ml_ai()
            elif choice == '3':
                self.game.reset()
                self.play_vs_hybrid_ai()
            elif choice == '4':
                print("Merci d'avoir joué !")
                break
            else:
                print("Option invalide")


if __name__ == "__main__":
    game_interface = GameInterface()
    game_interface.main_menu()