# Adversarial Search Connect Four Assignment
# Author: Mohamed Saeed
# Date: 2025-03-17
# Purpose: This program implements a Connect Four game (6 rows x 5 columns) where a human
# plays against an AI agent. The AI uses the minimax algorithm with alpha-beta pruning.


import time
import math
import random
import copy

# Board dimensions and game constants
ROWS = 6
COLS = 5
HUMAN_PIECE = 1
AI_PIECE = 2
EMPTY = 0

# Global counters for performance metrics
nodes_visited = 0
max_depth_reached = 0

def create_board():
    """
    Create an empty Connect Four board.
    Returns a 6x5 grid where:
    - 0 = empty space
    - 1 = human piece
    - 2 = AI piece
    """
    # Create empty board
    board = []
    for row in range(ROWS):
        # Each row has 5 empty spaces
        row = [EMPTY] * COLS
        board.append(row)
    return board

def print_board(board): 
    
    # Print each row with cell contents
    for row in board:
        
        cells = []
        for cell in row:
            if cell == EMPTY:
                cells.append(' ')    # Empty space
            elif cell == HUMAN_PIECE:
                cells.append('1')    # Human piece
            else:
                cells.append('2')    # AI piece
        
        # Print row with separators
        print('| ' + ' | '.join(cells) + ' |')
 
    
    # Print column numbers at the bottom
    print('  1   2   3   4   5  ')  # Column numbers
    print('(Choose 1-5 to play)')    # Helpful reminder

def is_valid_move(board, col):
    """
    Check if a move in this column is allowed.
    Returns True if the top cell in the column is empty.
    """
    return board[0][col] == EMPTY

def get_valid_moves(board):
    """
    Get a list of all columns where a move is possible.
    Returns a list of column numbers (0-4) that aren't full.
    """
    valid_columns = []
    for col in range(COLS):
        if is_valid_move(board, col):
            valid_columns.append(col)
    return valid_columns

def make_move(board, col, piece):
    """
    Drop a piece into the chosen column.
    The piece falls to the lowest empty spot in that column.
    
    Example:
    Before:     After:
    | | | | | |  | | | | | |
    | | | | | |  | | | | | |
    | | | | | |  | | | | | |
    | | | | | |  | | | | | |
    | | | | | |  | | | | | |
    | | | | | |  | | |2| | |
    """
    # Make a copy of the board so we don't change the original
    new_board = copy.deepcopy(board)
    #deepcopy is a copy whose properties do not share the same references 
    #(point to the same underlying values) as those of the source object from which the copy was made
    for row in range(ROWS-1, -1, -1): #start at bottom and goes up5
        # If we find an empty cell in this column
        if new_board[row][col] == EMPTY:
            new_board[row][col] = piece
            break
    
    return new_board

def check_win(board, piece):
    """
    Check if a player has won by getting 4 pieces in a row.
    Checks in all directions: horizontal, vertical, and both diagonals.
    """
    # Check horizontal (left to right)
    for row in range(ROWS):
        for col in range(COLS - 3):
            # Check 4 cells in a row
            if (board[row][col] == piece and 
                board[row][col+1] == piece and 
                board[row][col+2] == piece and 
                board[row][col+3] == piece):
                return True

    # Check vertical (top to bottom)
    for col in range(COLS):
        for row in range(ROWS - 3):
            # Check 4 cells in a column
            if (board[row][col] == piece and 
                board[row+1][col] == piece and 
                board[row+2][col] == piece and 
                board[row+3][col] == piece):
                return True

    # Check diagonal (top-left to bottom-right)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            # Check 4 cells diagonally down-right
            if (board[row][col] == piece and 
                board[row+1][col+1] == piece and 
                board[row+2][col+2] == piece and 
                board[row+3][col+3] == piece):
                return True

    # Check diagonal (bottom-left to top-right)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            # Check 4 cells diagonally up-right
            if (board[row][col] == piece and 
                board[row-1][col+1] == piece and 
                board[row-2][col+2] == piece and 
                board[row-3][col+3] == piece):
                return True

    return False

def check_tie(board):
    """Return True if the board is full and no moves remain."""
    return all(board[0][col] != EMPTY for col in range(COLS))

def evaluate_window(window, piece):
    """
    Look at 4 cells in a row and give them a score.
    Higher score means better position for the current player.
    """
    score = 0
    
    # Figure out which piece is the opponent's
    if piece == AI_PIECE:
        opponent_piece = HUMAN_PIECE
    else:
        opponent_piece = AI_PIECE
    
    # Count how many pieces we have in this window
    our_pieces = window.count(piece)
    opponent_pieces = window.count(opponent_piece)
    empty_spaces = window.count(EMPTY)
    
    # Give points for good positions
    if our_pieces == 4:
        score += 100    # We won!
    elif our_pieces == 3 and empty_spaces == 1:
        score += 5      # We're about to win
    elif our_pieces == 2 and empty_spaces == 2:
        score += 2      # We have a good start
    
    # Take away points for bad positions
    if opponent_pieces == 3 and empty_spaces == 1:
        score -= 4      # Opponent is about to win - we should block this
    
    return score

def evaluate_board(board, piece):
    """
    Look at the whole board and give it a score.
    Higher score means better position for the current player.
    
    Scoring considers:
    - Center column (more valuable)
    - Horizontal lines
    - Vertical lines
    - Both diagonal directions
    """
    score = 0

    # Center column is more valuable
    center_col = COLS // 2
    center_array = [board[row][center_col] for row in range(ROWS)]
    score += center_array.count(piece) * 3

    # Check horizontal lines
    for row in range(ROWS):
        for col in range(COLS - 3):
            window = board[row][col:col+4]
            score += evaluate_window(window, piece)

    # Check vertical lines
    for col in range(COLS):
        col_array = [board[row][col] for row in range(ROWS)]
        for row in range(ROWS - 3):
            window = col_array[row:row+4]
            score += evaluate_window(window, piece)

    # Check diagonal lines (top-left to bottom-right)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            window = [board[row+i][col+i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Check diagonal lines (bottom-left to top-right)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            window = [board[row-i][col+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board):
    """
    Check if the game is over.
    Returns True if:
    - Human player won
    - AI player won
    - Board is full (tie)
    """
    return (check_win(board, HUMAN_PIECE) or 
            check_win(board, AI_PIECE) or 
            check_tie(board))

def minimax(board, depth, alpha, beta, maximizingPlayer, start_time, time_limit, version):
    """
    Find the best move using the minimax algorithm with alpha-beta pruning.
    
    Parameters:
    - board: current game board
    - depth: how many moves ahead to look
    - alpha, beta: for pruning (ignore branches that won't help)
    - maximizingPlayer: True if it's AI's turn
    - start_time: when we started thinking
    - time_limit: how long we can think
    - version: 1 for basic, 2 for improved move ordering
    
    Returns: (score, best_move)
    """
    global nodes_visited, max_depth_reached
    nodes_visited += 1
    max_depth_reached = max(max_depth_reached, depth)
    
    # Stop if we're out of time
    if time.time() - start_time > time_limit:
        return evaluate_board(board, AI_PIECE), None

    # Stop if we've reached max depth or game is over
    if depth == 0 or is_terminal_node(board):
        if check_win(board, AI_PIECE):
            return (math.inf, None)      # AI wins
        elif check_win(board, HUMAN_PIECE):
            return (-math.inf, None)     # Human wins
        else:
            return (evaluate_board(board, AI_PIECE), None)  # Game not over

    # Get all possible moves
    valid_moves = get_valid_moves(board)
    
    # For version 2, sort moves by how good they look
    if version == 2:
        move_scores = []
        for move in valid_moves:
            temp_board = make_move(board, move, AI_PIECE if maximizingPlayer else HUMAN_PIECE)
            score = evaluate_board(temp_board, AI_PIECE)
            move_scores.append((score, move))
        if maximizingPlayer:
            valid_moves = [move for score, move in sorted(move_scores, key=lambda x: x[0], reverse=True)]
        else:
            valid_moves = [move for score, move in sorted(move_scores, key=lambda x: x[0])]

    # Pick a random move in case we don't find a better one
    best_move = random.choice(valid_moves)

    if maximizingPlayer:
        # AI's turn - try to maximize score
        value = -math.inf
        for move in valid_moves:
            child_board = make_move(board, move, AI_PIECE)
            new_score, _ = minimax(child_board, depth-1, alpha, beta, False, start_time, time_limit, version)
            if new_score > value:
                value = new_score
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # We can ignore the rest of the moves
        return value, best_move
    else:
        # Human's turn - try to minimize score
        value = math.inf
        for move in valid_moves:
            child_board = make_move(board, move, HUMAN_PIECE)
            new_score, _ = minimax(child_board, depth-1, alpha, beta, True, start_time, time_limit, version)
            if new_score < value:
                value = new_score
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break  # We can ignore the rest of the moves
        return value, best_move

def get_ai_move(board, version, time_limit):
    """
    Figure out the AI's next move.
    
    Parameters:
    - board: current game board
    - version: 1 for basic, 2 for improved move ordering
    - time_limit: how long we can think (in seconds)
    
    Returns:
    - best_move: the column to play in (0-4)
    - nodes_visited: how many positions we looked at
    - max_depth_reached: how far ahead we looked
    - total_time: how long we spent thinking
    """
    start_time = time.time()
    best_move = None
    depth = 1
    global nodes_visited, max_depth_reached
    nodes_visited = 0
    max_depth_reached = 0

    # Keep looking deeper until we run out of time
    while True:
        if time.time() - start_time > time_limit:
            break
            
        # Try this depth
        score, move = minimax(board, depth, -math.inf, math.inf, True, start_time, time_limit, version)
        
        # Only use this move if we finished thinking in time
        if time.time() - start_time < time_limit:
            best_move = move
            
        depth += 1
        
    total_time = time.time() - start_time
    return best_move, nodes_visited, max_depth_reached, total_time

def demonstrate_board_structure():
    """Demonstrate how the board is structured in memory."""
    board = create_board()
    print("\nBoard Structure Demonstration:")
    print("=============================")
    print("The board is a 2D array (6x5) where:")
    print("- 0 represents an empty cell")
    print("- 1 represents a human piece")
    print("- 2 represents an AI piece")
    print("\nEmpty board representation:")
    print("---------------------------")
    print("Array structure:")
    for row in board:
        print(row)
    print("\nVisual representation:")
    print_board(board)
    print("\nNote: The board is stored as a list of lists, where:")
    print("- The outer list contains 6 rows")
    print("- Each inner list contains 5 columns")
    print("- board[row][col] accesses the cell at position (row, col)")
    print("- Rows are numbered from top (0) to bottom (5)")
    print("- Columns are numbered from left (0) to right (4)")

def main():
    """
    Run the Connect Four game.
    - Human plays against AI
    - AI uses minimax algorithm
    - Game continues until someone wins or board is full
    """
    print("Welcome to Connect Four!")
    print("Board size: 6 rows x 5 columns. Get 4 in a row to win!")
    
    # Show how the board works
    demonstrate_board_structure()
    
    # Let player choose AI version
    while True:
        version_input = input("Select AI version (1 for basic, 2 for improved move ordering): ")
        if version_input in ['1', '2']:
            version = int(version_input)
            break
        else:
            print("Invalid input. Please enter 1 or 2.")

    # Set up the game
    board = create_board()
    game_over = False
    turn = 0  # 0 = human's turn, 1 = AI's turn

    # Main game loop
    while not game_over:
        print_board(board)
        
        if turn == 0:
            # Human's turn
            valid_move = False
            while not valid_move:
                try:
                    col = int(input("Your move (choose a column 1-5): ")) - 1
                    if col < 0 or col >= COLS:
                        print("Invalid column. Choose a column between 1 and 5.")
                    elif not is_valid_move(board, col):
                        print("Column is full. Choose another column.")
                    else:
                        valid_move = True
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 5.")
            
            # Make human's move
            board = make_move(board, col, HUMAN_PIECE)
            
            # Check if human won
            if check_win(board, HUMAN_PIECE):
                print_board(board)
                print("Congratulations! You win!")
                game_over = True
            elif check_tie(board):
                print_board(board)
                print("It's a tie!")
                game_over = True
                
        else:
            # AI's turn
            print("AI is thinking...")
            ai_time_limit = 1  # seconds per move (This still somehow beats me idk how, I think my pc processor is too strong)
            
            # Get AI's move
            move, nodes, depth_reached, time_taken = get_ai_move(board, version, ai_time_limit)
            
            if move is None:
                print("AI could not determine a move in time. It's a tie!")
                game_over = True
            else:
                # Make AI's move
                board = make_move(board, move, AI_PIECE)
                print(f"AI played in column {move+1}.")
                print(f"AI performance: Nodes evaluated = {nodes}, Max depth reached = {depth_reached}, Time taken = {time_taken:.2f} seconds")
                
                # Check if AI won
                if check_win(board, AI_PIECE):
                    print_board(board)
                    print("AI wins! Better luck next time.")
                    game_over = True
                elif check_tie(board):
                    print_board(board)
                    print("It's a tie!")
                    game_over = True
        
        # Switch turns
        turn = (turn + 1) % 2

if __name__ == "__main__":
    main()
