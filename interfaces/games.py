import tkinter as tk
from tkinter import messagebox, font
import joblib
import numpy as np
from PIL import Image, ImageTk
import os

# Charger modèles ML
try:
    model_x = joblib.load("../models/model_xwins.pkl")
    model_d = joblib.load("../models/model_draw.pkl")
    models_loaded = True
except:
    models_loaded = False
    print("Attention: Modèles ML non trouvés")

# Plateu

def print_board(board):
    symbols = {1: "X", -1: "O", 0: " "}
    for i in range(0, 9, 3):
        print("|".join(symbols[board[i+j]] for j in range(3)))
        print("-"*5)

def check_winner(board):
    wins = [(0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)]
    
    for a,b,c in wins:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0

def is_full(board):
    return 0 not in board

def encode(board):
    features = []
    for cell in board:
        features.append(1 if cell == 1 else 0)
        features.append(1 if cell == -1 else 0)
    return features


# IA ML

def best_move_ml(board):
    if not models_loaded:
        # Heuristique simple si modèles non disponibles
        available = [i for i in range(9) if board[i] == 0]
        if available:
            return available[0]
        return None
        
    best_score = -999
    move = None

    for i in range(9):
        if board[i] == 0:
            board[i] = 1  # X joue

            features = encode(board)
            proba_win = model_x.predict_proba([features])[0][1]
            proba_draw = model_d.predict_proba([features])[0][1]

            score = proba_win + 0.5 * proba_draw

            board[i] = 0

            if score > best_score:
                best_score = score
                move = i

    return move


# IA Hybride (Minimax + ML)

def minimax_hybrid(board, depth, is_x_turn):
    winner = check_winner(board)

    if winner == 1:
        return 1
    elif winner == -1:
        return -1
    elif is_full(board):
        return 0

    # profondeur limite → utiliser ML
    if depth == 0:
        if models_loaded:
            features = encode(board)
            proba_win = model_x.predict_proba([features])[0][1]
            proba_draw = model_d.predict_proba([features])[0][1]
            return proba_win - (1 - proba_win - proba_draw)
        else:
            return 0

    if is_x_turn:
        best = -999
        for i in range(9):
            if board[i] == 0:
                board[i] = 1
                score = minimax_hybrid(board, depth-1, False)
                board[i] = 0
                best = max(best, score)
        return best
    else:
        best = 999
        for i in range(9):
            if board[i] == 0:
                board[i] = -1
                score = minimax_hybrid(board, depth-1, True)
                board[i] = 0
                best = min(best, score)
        return best

def best_move_hybrid(board):
    best_score = -999
    move = None

    for i in range(9):
        if board[i] == 0:
            board[i] = 1
            score = minimax_hybrid(board, 3, False)
            board[i] = 0

            if score > best_score:
                best_score = score
                move = i

    return move


# Interface Graphique Stylisée


class TicTacToeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Morpion IA - Tic Tac Toe")
        self.root.geometry("600x750")
        self.root.resizable(False, False)
        
        # Couleurs et styles
        self.colors = {
            'bg': '#2C3E50',
            'board': '#34495E',
            'cell': '#ECF0F1',
            'x_color': '#E74C3C',
            'o_color': '#3498DB',
            'button': '#27AE60',
            'button_hover': '#229954',
            'button_secondary': '#E67E22',
            'button_danger': '#E74C3C',
            'text': '#FFFFFF'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Police personnalisée
        self.title_font = font.Font(family="Helvetica", size=24, weight="bold")
        self.cell_font = font.Font(family="Helvetica", size=48, weight="bold")
        self.button_font = font.Font(family="Helvetica", size=12, weight="bold")
        
        self.board = [0] * 9
        self.current_player = 1  # 1 = X (IA), -1 = O (Humain)
        self.game_mode = None
        self.game_active = True
        self.ai_thinking = False  # Pour éviter les appels multiples
        
        # Interface utilisateur
        self.setup_ui()
        
    def setup_ui(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Titre
        title_label = tk.Label(main_frame, text="MORPION", 
                               font=self.title_font, 
                               bg=self.colors['bg'], 
                               fg=self.colors['text'])
        title_label.pack(pady=(0, 20))
        
        # Menu de sélection
        self.menu_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        self.menu_frame.pack(expand=True)
        
        # Boutons du menu
        menu_buttons = [
            ("Humain vs Humain", "human"),
            ("vs IA (ML)", "ml"),
            ("vs IA (Hybride)", "hybrid")
        ]
        
        for text, mode in menu_buttons:
            btn = tk.Button(self.menu_frame, text=text, 
                          font=self.button_font,
                          bg=self.colors['button'],
                          fg=self.colors['text'],
                          activebackground=self.colors['button_hover'],
                          activeforeground=self.colors['text'],
                          bd=0, padx=30, pady=15,
                          cursor="hand2",
                          command=lambda m=mode: self.start_game(m))
            btn.pack(pady=10)
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=self.colors['button_hover']))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=self.colors['button']))
        
        # Plateau de jeu (caché initialement)
        self.board_frame = tk.Frame(main_frame, bg=self.colors['board'])
        self.buttons = []
        
        for i in range(9):
            btn = tk.Button(self.board_frame, text="", 
                          font=self.cell_font,
                          width=4, height=2,
                          bg=self.colors['cell'],
                          activebackground=self.colors['cell'],
                          bd=2,
                          relief="solid",
                          cursor="hand2",
                          command=lambda idx=i: self.make_move(idx))
            btn.grid(row=i//3, column=i%3, padx=3, pady=3, sticky="nsew")
            self.buttons.append(btn)
        
        # Configurer les poids des grilles
        for i in range(3):
            self.board_frame.grid_rowconfigure(i, weight=1)
            self.board_frame.grid_columnconfigure(i, weight=1)
        
        # Frame d'information
        self.info_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        
        self.status_label = tk.Label(self.info_frame, text="", 
                                    font=self.button_font,
                                    bg=self.colors['bg'],
                                    fg=self.colors['text'])
        self.status_label.pack(pady=10)
        
        # Frame pour les boutons de contrôle
        control_frame = tk.Frame(self.info_frame, bg=self.colors['bg'])
        control_frame.pack(pady=10)
        
        # Bouton Rejouer
        self.replay_button = tk.Button(control_frame, text="Rejouer",
                                     font=self.button_font,
                                     bg=self.colors['button'],
                                     fg=self.colors['text'],
                                     activebackground=self.colors['button_hover'],
                                     bd=0, padx=20, pady=10,
                                     cursor="hand2",
                                     command=self.replay_game)
        self.replay_button.pack(side=tk.LEFT, padx=5)
        
        # Bouton Retour au Menu
        self.back_button = tk.Button(control_frame, text="Retour Menu",
                                   font=self.button_font,
                                   bg=self.colors['button_secondary'],
                                   fg=self.colors['text'],
                                   activebackground='#D35400',
                                   bd=0, padx=20, pady=10,
                                   cursor="hand2",
                                   command=self.back_to_menu)
        self.back_button.pack(side=tk.LEFT, padx=5)
        
        # Bouton Quitter
        self.quit_button = tk.Button(self.info_frame, text="Quitter",
                                   font=self.button_font,
                                   bg=self.colors['button_danger'],
                                   fg=self.colors['text'],
                                   activebackground='#C0392B',
                                   bd=0, padx=20, pady=10,
                                   cursor="hand2",
                                   command=self.quit_game)
        self.quit_button.pack(pady=5)
        
        # Ajouter les effets de survol
        self.replay_button.bind("<Enter>", lambda e: self.replay_button.configure(bg=self.colors['button_hover']))
        self.replay_button.bind("<Leave>", lambda e: self.replay_button.configure(bg=self.colors['button']))
        self.back_button.bind("<Enter>", lambda e: self.back_button.configure(bg='#D35400'))
        self.back_button.bind("<Leave>", lambda e: self.back_button.configure(bg=self.colors['button_secondary']))
        self.quit_button.bind("<Enter>", lambda e: self.quit_button.configure(bg='#C0392B'))
        self.quit_button.bind("<Leave>", lambda e: self.quit_button.configure(bg=self.colors['button_danger']))
        
    def start_game(self, mode):
        self.game_mode = mode
        self.ai_thinking = False
        self.reset_board()
        self.menu_frame.pack_forget()
        self.board_frame.pack(expand=True, fill='both', pady=20)
        self.info_frame.pack(pady=10)
        
        # Configuration selon le mode
        if mode == "human":
            self.current_player = 1  # X commence
            self.update_status("Joueur X (⚡) commence !")
        elif mode == "ml":
            # L'IA joue X, l'humain joue O
            self.current_player = 1  # C'est le tour de l'IA (X)
            self.update_status("L'IA réfléchit... 🤔")
            # L'IA commence en premier
            self.root.after(500, self.ai_move)
        elif mode == "hybrid":
            # L'IA joue X, l'humain joue O
            self.current_player = 1  # C'est le tour de l'IA (X)
            self.update_status("L'IA réfléchit... 🤔")
            # L'IA commence en premier
            self.root.after(500, self.ai_move)
        
        self.update_board_display()
    
    def replay_game(self):
        """Fonction pour rejouer une partie avec le même mode"""
        if self.ai_thinking:
            return
        
        # Réinitialiser le jeu
        self.reset_board()
        self.ai_thinking = False
        self.game_active = True
        
        # Réactiver tous les boutons
        for btn in self.buttons:
            btn.configure(state="normal")
        
        # Recommencer selon le mode
        if self.game_mode == "human":
            self.current_player = 1
            self.update_status("Joueur X (⚡) commence !")
        elif self.game_mode == "ml":
            self.current_player = 1
            self.update_status("L'IA réfléchit... 🤔")
            self.root.after(500, self.ai_move)
        elif self.game_mode == "hybrid":
            self.current_player = 1
            self.update_status("L'IA réfléchit... 🤔")
            self.root.after(500, self.ai_move)
        
        self.update_board_display()
        
    def reset_board(self):
        self.board = [0] * 9
        self.game_active = True
        self.ai_thinking = False
        
        for btn in self.buttons:
            btn.configure(text="", state="normal", bg=self.colors['cell'])
    
    def back_to_menu(self):
        """Retour au menu principal"""
        if self.ai_thinking:
            return
        
        # Cacher l'interface de jeu
        self.board_frame.pack_forget()
        self.info_frame.pack_forget()
        
        # Afficher le menu
        self.menu_frame.pack(expand=True)
        
        # Réinitialiser complètement
        self.reset_board()
        self.game_mode = None
        self.ai_thinking = False
    
    def quit_game(self):
        if messagebox.askyesno("Quitter", "Voulez-vous vraiment quitter le jeu ?"):
            self.root.quit()
    
    def update_status(self, message):
        self.status_label.configure(text=message)
        self.root.update()
    
    def update_board_display(self):
        symbols = {1: "X", -1: "O", 0: ""}
        for i, value in enumerate(self.board):
            color = self.colors['x_color'] if value == 1 else self.colors['o_color'] if value == -1 else self.colors['cell']
            self.buttons[i].configure(text=symbols[value], fg=color)
    
    def make_move(self, position):
        # Vérifier si le jeu est actif et si c'est le tour du joueur humain
        if not self.game_active or self.ai_thinking:
            return
        
        # Dans les modes IA, l'humain joue O (current_player = -1)
        if self.game_mode in ["ml", "hybrid"] and self.current_player != -1:
            messagebox.showinfo("Au tour de l'IA", "Attendez que l'IA joue !")
            return
        
        if self.board[position] != 0:
            messagebox.showwarning("Case occupée", "Cette case est déjà prise !")
            return
        
        # Faire le mouvement
        self.board[position] = self.current_player
        self.update_board_display()
        
        # Vérifier fin de partie
        winner = check_winner(self.board)
        if winner != 0:
            self.end_game(winner)
            return
        elif is_full(self.board):
            self.end_game(0)
            return
        
        # Changer de joueur
        self.current_player *= -1
        
        # Si c'est le tour de l'IA dans les modes IA
        if self.game_active:
            if self.game_mode == "human":
                player_name = "X (⚡)" if self.current_player == 1 else "O (🔵)"
                self.update_status(f"Au tour de {player_name}")
            elif self.game_mode in ["ml", "hybrid"] and self.current_player == 1:
                # C'est le tour de l'IA
                self.update_status("L'IA réfléchit... 🤔")
                self.ai_thinking = True
                self.root.after(500, self.ai_move)
    
    def ai_move(self):
        if not self.game_active or self.current_player != 1:
            self.ai_thinking = False
            return
        
        # Faire le mouvement de l'IA
        if self.game_mode == "ml":
            move = best_move_ml(self.board.copy())
        else:  # hybrid
            move = best_move_hybrid(self.board.copy())
        
        if move is not None and self.board[move] == 0:
            self.board[move] = self.current_player
            self.update_board_display()
            
            # Vérifier fin de partie
            winner = check_winner(self.board)
            if winner != 0:
                self.end_game(winner)
                self.ai_thinking = False
                return
            elif is_full(self.board):
                self.end_game(0)
                self.ai_thinking = False
                return
            
            # Passer le tour à l'humain
            self.current_player = -1
            self.update_status("Votre tour ! (O)")
        
        self.ai_thinking = False
    
    def end_game(self, winner):
        self.game_active = False
        self.ai_thinking = False
        
        if winner == 1:
            if self.game_mode == "human":
                message = "🎉 Joueur X (⚡) remporte la partie ! 🎉"
            else:
                message = "🤖 L'IA (X) a gagné ! 🤖"
            self.update_status(message)
            messagebox.showinfo("Partie terminée", message + "\n\nCliquez sur 'Rejouer' pour une nouvelle partie")
        elif winner == -1:
            if self.game_mode == "human":
                message = "🎉 Joueur O (🔵) remporte la partie ! 🎉"
            else:
                message = "🎉 Félicitations ! Vous (O) avez gagné ! 🎉"
            self.update_status(message)
            messagebox.showinfo("Partie terminée", message + "\n\nCliquez sur 'Rejouer' pour une nouvelle partie")
        else:
            message = "🤝 Match nul ! Bien joué ! 🤝"
            self.update_status(message)
            messagebox.showinfo("Partie terminée", message + "\n\nCliquez sur 'Rejouer' pour une nouvelle partie")
        
        # Désactiver les boutons du plateau
        for btn in self.buttons:
            btn.configure(state="disabled")
    
    def run(self):
        self.root.mainloop()


# Lancement de l'application


def main():
    # Vérifier si les modèles sont chargés
    if not models_loaded:
        print("\n" + "="*50)
        print("⚠️  ATTENTION: Modèles ML non trouvés")
        print("Les modes IA utiliseront des heuristiques simples")
        print("Placez vos modèles dans le dossier 'models/'")
        print("Fichiers attendus: model_xwins.pkl, model_draw.pkl")
        print("="*50 + "\n")
    
    app = TicTacToeGUI()
    app.run()

if __name__ == "__main__":
    main()