import chess
import os
from glicko2 import Glicko2, WIN, DRAW, LOSS
import pandas as pd
import random
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = 
chess_startgame = """
Start:

1. Ruy Lopez (Spanish Opening):

Move order: 1.e4 e5 2.Nf3 Nc6 3.Bb5

Key ideas: Control the center, pressure Black's e5 pawn, prepare castling

Variations: Berlin Defense, Marshall Attack, Closed Variation

2. Sicilian Defense:

Move order: 1.e4 c5

Key ideas: Fight for d4 square, create imbalanced positions

Variations: Najdorf, Dragon, Scheveningen, Classical

3. Queen's Gambit:

Move order: 1.d4 d5 2.c4

Key ideas: Control the center, develop pieces quickly

Variations: Accepted, Declined, Slav Defense

4. King's Indian Defense:

Move order: 1.d4 Nf6 2.c4 g6

Key ideas: Hypermodern approach, control the center from afar

Variations: Classical, Sämisch, Four Pawns Attack

5. French Defense:

Move order: 1.e4 e6

Key ideas: Solid structure, counterattack in the center

Variations: Winawer, Classical, Tarrasch
"""

chess_middlegame = """
Middle Game

1. Pawn Structures:

Isolated Queen's Pawn: Control key squares, prepare minority attack

Hanging Pawns: Maintain tension, avoid exchanges that weaken structure

Pawn Chains: Attack at the base, play on the side where you have space advantage

2. Piece Coordination:

Rooks on Open Files: Double rooks, control open files

Bishop Pair: Maintain both bishops, create threats on long diagonals

Knight Outposts: Establish knights in enemy territory, supported by pawns

3. King Safety:

Fianchettoed Bishop: Maintain the fianchetto, avoid weakening pawn structure

Castled Position: Avoid pawn moves in front of the king, maintain pawn shield

Opposite Side Castling: Launch pawn storm, open lines for attack

4. Strategic Concepts:

Space Advantage: Restrict opponent's pieces, prepare pawn breaks

Weak Squares: Create and exploit weak squares in enemy position

Piece Activity: Activate pieces, control key squares and diagonals

5. Tactical Motifs:

Pin: Create and exploit pins, especially against king or undefended pieces

Fork: Set up knight forks, discover opportunities for double attacks

Discovered Attack: Prepare powerful discovered checks or attacks
"""

chess_endgame = """
End Game

1. Pawn Endgames:

Opposition: Understand and use opposition to promote pawns

Pawn Breakthrough: Calculate pawn breaks to create passed pawns

Key Squares: Identify and control key squares for pawn promotion

2. Rook Endgames:

Lucena Position: Learn the winning technique for rook and pawn vs rook

Philidor Position: Understand defensive techniques in rook endgames

Rook Behind Passed Pawn: Place rook behind passed pawns (yours or opponent's)

3. Minor Piece Endgames:

Bishop vs Knight: Exploit the strengths of bishop in open positions

Same-Colored Bishops: Understand drawing techniques and winning chances

Knight Endgames: Centralize knight, create and exploit weaknesses

4. Queen Endgames:

Queen vs Pawn: Learn winning techniques and drawing fortresses

Queen vs Rook: Understand skewer and stalemate motifs

5. Theoretical Positions:

Vancura Position: Defensive technique in rook vs rook and pawn

Troitzky Line: Winning positions in two knights vs pawn endgames

König's Technique: Winning method in queen vs rook endgames
"""

comprehensive_chess_knowledge = chess_startgame + chess_middlegame + chess_endgame

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(comprehensive_chess_knowledge)

# Set up embeddings and vector store
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])


# Set up retriever
retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# Set up QA chain
llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def calculate_elo_change2(player_rating, opponent_rating, result):
    if result == 1:
        new_player_rating, new_opponent_rating = env.rate_1vs1(player_rating, opponent_rating, drawn=False)
    elif result == 0.5:
        new_player_rating, new_opponent_rating = env.rate_1vs1(player_rating, opponent_rating, drawn=True)
    else:  # result == 0
        new_opponent_rating, new_player_rating = env.rate_1vs1(opponent_rating, player_rating, drawn=False)
    
    return new_player_rating, new_opponent_rating

def calculate_elo_change(player_rating, opponent_rating, result, k_factor=32):
    """
    Calculate the change in ELO rating after a game.

    Parameters:
    - player_rating (float): The ELO rating of the player.
    - opponent_rating (float): The ELO rating of the opponent.
    - result (float): The result of the game (1 for win, 0.5 for draw, 0 for loss).
    - k_factor (int, optional): The development coefficient. Defaults to 32.

    Returns:
    - float: The change in ELO rating.
    """
    expected_score = 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))
    elo_change = k_factor * (result - expected_score)
    new_player_rating = player_rating + elo_change
    new_opponent_rating = opponent_rating - elo_change
    return new_player_rating, new_opponent_rating

def load_puzzles(filename="lichess_db_puzzle.csv"):
    """
    Load puzzles from a CSV file.
    
    Returns: DataFrame containing puzzles.
    """
    return pd.read_csv(filename)

def select_random_puzzles(puzzles):
    """
    Select a random puzzle from the DataFrame. Returns: A single puzzle as a Series.
    """
    sorted_puzzles = puzzles.assign(SortKey=puzzles['NbPlays'] * puzzles['Popularity']).sort_values(by='SortKey', ascending=False).head(100000)
    return sorted_puzzles.sample(n=1000)

def solve_puzzle_with_gpt(puzzle):
    """
    Attempt to solve a selected chess puzzle using GPT.
    
    puzzle: A Series containing puzzle information.
    """
    board = chess.Board(puzzle['FEN'])
    moves = puzzle['Moves'].split()
    
    #print(f"Solving puzzle: {puzzle['PuzzleId']} with FEN: {puzzle['FEN']}")
    # Play the first move on the board
    first_move = moves[0]
    #print(f"Playing first move: {first_move}")
    board.push_uci(first_move)
    # Now ask GPT to solve for the next move
    move_uci = gpt_chess_move(board, 'white' if board.turn == chess.WHITE else 'black')
    if not move_uci:
        return -1
    if len(moves) > 1 and move_uci != moves[1]:
        #print(f"GPT failed to solve the puzzle. Expected {moves[1]} but got {move_uci}")
        return 0
    else:
        #print("GPT successfully predicted the next move!")
        return 1



def gpt_chess_move(board, color):
    """
    Ask GPT for a chess move.
    board: chess.Board object representing the current game state.
    color: 'white' or 'black' indicating which player's move to generate.
    
    Returns: A move in UCI format (e.g., 'e2e4') suggested by GPT-4.
    """
    prompt = f"The chess board is in the following state (FEN): '{board.fen()}'. Give the move you want to use for {color}. You are allowed to answer only with the move itself and nothing else. You must try to give a move."
    #print(prompt)
    try:
        response = qa.run(query=prompt)
        print(response)
        text_response = response.lstrip('.')
        move_san = text_response.split()[0]  # Taking the first word as the move
        move = board.parse_san(move_san)  # Converts SAN to move object
        return move.uci()  # Converts move object to UCI format
    except Exception as e:
        #print(f"Error during GPT query or parsing: {e}")
        return None


def play_chess_with_gpt():
    board = chess.Board()

    while not board.is_game_over():
        color = 'white' if board.turn == chess.WHITE else 'black'
        move_uci = gpt_chess_move(board, color)
        
        if move_uci and chess.Move.from_uci(move_uci) in board.legal_moves:
            move = chess.Move.from_uci(move_uci)
            board.push(move)
            print(f"{color.capitalize()}'s move: {move_uci}")
            print(board)
        else:
            print(f"Received an illegal or no move from GPT for {color}: {move_uci}")
            break

    print("Game over. Result:", board.result())


# this is a mode to just have models play chess in real time
#if __name__ == "__main__":
    #model1=llms.init('gpt-4-turbo-preview')
    #model2=llms.init('gpt-4')
    #play_chess_with_gpt()
    
if __name__ == "__main__":
   
    # Code to randomly select puzzles from Lichess dataset
    #puzzles = load_puzzles()
    #puzzles=select_random_puzzles(puzzles)
    #puzzles.to_csv('puzzles.csv', index=False)

    puzzles=pd.read_csv('puzzles.csv')
    
    # models=['gpt-3.5-turbo','gpt-4-turbo-preview','gpt-4','claude-3-opus-20240229','mistral-large-latest','open-mixtral-8x7b','claude-3-sonnet-20240229','claude-3-haiku-20240307','claude-instant-1.2','gemini-1.5-pro-latest']
    # models=[ 'gemini-1.5-pro-latest']
    model = ['GPT-4-RAG']
    env = Glicko2(tau=0.5)

    r1 = env.create_rating(1000, 400, 0.06) # assumed start elo rating

    count_good=0
    count_illegal=0
    score=0
    wins=0
    losses=0
    
    streak=[]
    solved=[]
    elo=0
    elo2=0
    
    for index, puzzle in puzzles.iterrows():
    #for index, puzzle in puzzles.head(1000).iterrows():
        print(f"Solving puzzle {index + 1}:")
        rating=puzzle['Rating']
        r2=env.create_rating(rating, puzzle['RatingDeviation'])
        result = solve_puzzle_with_gpt(puzzle)#random.randint(0,1)#
        if result>0: #puzzle solved
            count_good+=1
            wins+=1
            elo, opp=calculate_elo_change(elo, rating, 1)
            elo2, opp=calculate_elo_change2(r1, r2, 1)
            streak.append((WIN,r2))
            solved.append(rating)

            print(f"Puzzle {index + 1} ({rating}) solved. Try it: {puzzle['GameUrl']} Score: {score} Elo: {int(env.rate(r1, streak).mu)} adjusted:{int(env.rate(r1, streak).mu*(1-count_illegal/(wins+losses)))}")
        else:
            if result<0: # if -1 then GPT made illegal move
                count_illegal+=1
                if 0: # we can further penalize it
                    losses+=1
                    r3=env.create_rating(rating/2, puzzle['RatingDeviation'])
                    elo, opp=calculate_elo_change(elo, rating, 0)
                    elo2, opp=calculate_elo_change2(r1, r3, 0)
                    streak.append((LOSS,r2))
                
            losses+=1
            elo, opp=calculate_elo_change(elo, rating, 0)
            elo2, opp=calculate_elo_change2(r1, r2, 0)
            streak.append((LOSS,r2))
            
        score+=result
    rated = env.rate(r1, streak)
    print(f"{model} Score: {score} Solved: {count_good} Illegal: {count_illegal} Elo: {int(rated.mu)} adjusted:{int(env.rate(r1, streak).mu*(1-count_illegal/(wins+losses)))} ")
 
