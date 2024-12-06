import os
import llms
from difflib import SequenceMatcher

model1 = llms.init(
    openai_api_key='your_api_key', 
    model='gpt-3.5-turbo')
model2  = llms.init(
    openai_api_key='your_api_key', 
    model='gpt-3.5-turbo')

class Gomoku:
    def __init__(self):
        self.board = [[' ' for _ in range(15)] for _ in range(15)]
        self.is_over = False
        self.winner = None
        self.current_player = 1  # Initialize current_player
        self.history = []  # Initialize history to track moves

    def __str__(self):
        board_str = "+".join(["-" * 3 for _ in range(15)]) + "\n"
        for row in self.board:
            board_str += "|".join([" " + col + " " for col in row]) + "\n"
        return board_str

    def get_board_notation(self):
        notation = []
        for row in self.board:
            notation.append("".join(['X' if cell == 'X' else 'O' if cell == 'O' else '.' for cell in row]))
        return ",".join(notation)

    def is_game_over(self):
        return self.is_over

    def is_winner(self, player):
        return self.winner == player

    def is_valid_move(self, row, col):
        return 0 <= row < 15 and 0 <= col < 15 and self.board[row][col] == ' '

    def make_move(self, row, col):
        player_symbol = 'X' if self.current_player == 1 else 'O'
        self.board[row][col] = player_symbol
        self.history.append({"role": "user", "content": f"Player {player_symbol} plays at ({row}, {col}). Board state: {self.get_board_notation()}"})
        self.check_winner(row, col)

    def check_winner(self, row, col):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        player_symbol = self.board[row][col]
        
        for dr, dc in directions:
            count = 1
            for step in range(1, 5):
                r, c = row + step * dr, col + step * dc
                if 0 <= r < 15 and 0 <= c < 15 and self.board[r][c] == player_symbol:
                    count += 1
                else:
                    break

            for step in range(1, 5):
                r, c = row - step * dr, col - step * dc
                if 0 <= r < 15 and 0 <= c < 15 and self.board[r][c] == player_symbol:
                    count += 1
                else:
                    break

            if count >= 5:
                self.is_over = True
                self.winner = 1 if player_symbol == 'X' else 2
                return

        # Check for draw
        if all(self.board[r][c] != ' ' for r in range(15) for c in range(15)):
            self.is_over = True
            self.winner = None

def get_invalid_positions(game):
    """Get all invalid (occupied) positions on the board."""
    invalid_positions = []
    for row in range(15):
        for col in range(15):
            if game.board[row][col] != ' ':  # test validation
                invalid_positions.append((row, col))
    return invalid_positions

import random


def gpt_gomoku_move(game, model):
    """
    Ask GPT for a Gomoku move.
    game: Gomoku game object representing the current game state.
    model: The GPT model to use for generating the move.
    
    Returns: Tuple (row, column) where the move is to be made.
    """
    board_notation = game.get_board_notation()
    invalid_positions = get_invalid_positions(game)
    invalid_positions_str = ", ".join([f"({r}, {c})" for r, c in invalid_positions])

    system_prompt = f"""You are AI playing a game of Gomoku. The current board state is represented by rows from top to bottom and columns from left to right, using 'X' for black stones, 'O' for white stones, and '.' for empty cells. 

    The board state is: {board_notation}
    The following positions are invalid (already occupied): {invalid_positions_str}.
    Please select a move in the format <move>(row, column)</move>, avoiding the invalid positions.
    Do not use any other format.
    The goal is to connect five stones in any direction (horizontal, vertical, or diagonal).

    Example 1:
    Input: The board state is: ..............,...............,..............
    Player X to move.
    Output: <move>(7, 7)</move>

    Example 2:
    Input: The board state is: ..............,...............,..............
    Player O to move.
    Output: <move>(8, 8)</move>

    Remember, output must always be in the <move>(row, column)</move> format."""

    player_symbol = 'X' if game.current_player == 1 else 'O'
    prompt = f"The board state is: {board_notation}\nPlayer {player_symbol} to move. "
    
    for _ in range(3):  # try multiple times to avoid invalid response
        try:
            response = model.complete(prompt=prompt, system_message=system_prompt, max_tokens=500)
            print("GPT Response:", response.text)
            
            #  <move>(row, column)</move> 
            import re
            match = re.search(r'<move>\((\d+),\s*(\d+)\)</move>', response.text)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                if game.is_valid_move(row, col):  # validation
                    return row, col
                else:
                    print(f"Invalid move chosen by GPT: ({row}, {col}) - Position already occupied or out of bounds.")
            else:
                print("No valid move format found in GPT response.")
        except Exception as e:
            print(f"Error during GPT query or parsing: {e}")

    # If still fails after multiple tries
    print("Failed to get a valid move from GPT after multiple attempts.")
    return None

def play_gomoku_with_gpt():
    game = Gomoku()
   
    while not game.is_game_over():
        model = model1 if game.current_player == 1 else model2
        move = gpt_gomoku_move(game, model)
        
        if move is None:
            print(f"Failed to get a valid move for Player {game.current_player}. Ending the game.")
            break

        if game.is_valid_move(move[0], move[1]):
            game.make_move(move[0], move[1])
            print(f"Player {'X' if game.current_player == 1 else 'O'}'s move: {move}")
            print(game)
            game.current_player = 3 - game.current_player
        else:
            print(f"Received an illegal move from GPT for Player {game.current_player}: {move}")
    
    if game.is_winner(1):
        print("Player 1 wins!")
    elif game.is_winner(2):
        print("Player 2 wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    play_gomoku_with_gpt()