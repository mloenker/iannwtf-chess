import chess
import random

def extract_start(game, stop_move):
  """
  Extracts the game until a predefined move
    - game = game in SAN notation
    - stop_move = move number indicating end of extraction
  """
  #find beginning of stop move
  loc = game.find(' ' + str(stop_move) + '. ')
  #return opening
  if loc == -1:
    return game
  return game[:loc]

def generate_common_openings(num_games, training_data):
  """
  Finds common game openings (of training data) needed for model evaluation 1
    - num_games = amount of openings needed
    - training_data = all games used for training the model
  """
  openings = []
  for game in training_data:
    #get game opening, i.e. first and second move
    opening = extract_start(game=game, stop_move=3)
    #add opening 
    openings.append(opening)
  #count opening occurrences 
  occ = {i:openings.count(i) for i in openings}
  #select most common openings as startpositions
  eval_startpos = sorted(occ, key=occ.get, reverse=True)[:num_games]
  return eval_startpos

def generate_startpos(num_games, num_moves, testing_data):
  """
  Generates all start positions needed for model evaluation 2 and 3
    - num_games = amount of games on which model is tested
    - num_moves = amount of moves taken to get start position of game
    - testing_data = human games that the model has not seen yet
  """
  eval_startpos = []
  for game in testing_data[:num_games]:
    #get first moves of game
    opening = extract_start(game=game, stop_move=num_moves+1)
    #add start position
    eval_startpos.append(opening)
  return eval_startpos

def get_random_state(num_moves):
  """
  Generates a random game state after a predefined number of moves
    - num_moves = amount of moves taken
  """
  move_seq = ''
  #create a new chess board
  board = chess.Board()
  #generate random moves in SAN notation
  for i in range(num_moves):
    w_move = ''
    b_move = ''
    for player in range(2):
      #get all legal moves of current board state
      legal = list(board.legal_moves)
      #make random move + save its SAN
      move = random.choice(legal)
      if player == 0: #white
        w_move = board.san(move)
      else: #black
        b_move = board.san(move)
      board.push(move)
    #add move to move sequence
    move_seq = move_seq + f"{i+1}. " + w_move + ' ' + b_move + ' '
  return move_seq

def generate_random_startpos(num_games, num_moves):
  """
  Generates all start positions needed for model evaluation 4 and 5
    - num_games = amount of games on which model is tested
    - num_moves = amount of moves taken to get start position of game
  """
  eval_startpos = []
  for game in range(num_games):
    #generate and add random start position
    move_seq = get_random_state(num_moves)
    eval_startpos.append(move_seq)
  return eval_startpos
