import numpy as np
import matplotlib.pyplot as plt
import chess
from tabulate import tabulate
from create_eval_startpos import extract_start, generate_common_openings, generate_startpos, generate_random_state, generate_random_startpos
from create_eval_visual import create_graph, create_table

def prepare_startpos(seq):
  """
    Removes the move count of the input sequence and splits it into single moves
      - seq = single input sequence in SAN
  """
  #split sequence into moves
  moves_with_nr = seq.split()
  #remove move count
  moves = np.array(moves_with_nr)
  moves = np.delete(moves, np.arange(0, moves.size, 3))
  return moves_with_nr, moves

def generate_game(model, tokenizer, startpos, n_tokens=30):
  """
    Model generates a game of a predefined number of moves from a given startposition
      - model = model used to generate game
      - tokenizer = tokenizer used for encoding and decoding
      - startpos = single input sequence
      - n_tokens = amount of tokens that should be generated
  """
  #prepare prompt: convert list back into single string
  prompt = ' '.join(startpos)
  #generate game
  output = model.generate(prompt=prompt,max_new_tokens=n_tokens,num_return_sequences=1)
  #split game into moves
  output = output[0].split(" ")
  #remove startposition
  output = output[len(startpos):]
  #remove last move to avoid potential incompleteness of the moves
  idx = max(idx for idx, x in enumerate(output) if '.' in x)
  output = output[:idx]
  #remove move numbers
  game = [ x for x in output if "." not in x ]
  return game, output

def validation(n_valid, model, tokenizer, startpos, board, n_tokens=30):
  """
    Generates game until first illegal move, returns # of legal moves and illegal move in SAN
      - n_valid = current amount of valid moves generated
      - model = model used to generate game
      - tokenizer = tokenizer used for encoding and decoding
      - startpos = single input sequence leading to current board state in SAN
      - board = chess board object
      - n_tokens = amount of tokens that should be generated
  """
  valid = True
  #generate game 
  moves, movesWnr = generate_game(model, tokenizer, startpos)
  #check generated moves
  for move in moves:
    #check if move notation is legal
    try:
      p_move = str(board.parse_san(move))
    except (chess.IllegalMoveError, chess.AmbiguousMoveError, chess.InvalidMoveError):
      illegal_move = move
      type_illegal = "Wrong notation"
      valid = False
      break
    #find all legal moves in current position
    legal = [move.uci() for move in board.legal_moves]
    #check if move is possible at current board state
    if p_move in legal:
      n_valid += 1
      board.push_san(move)
    else:
      illegal_move = move
      type_illegal = "Wrong arrangement"
      valid = False
      break
  #generate + check more moves if necessary
  if valid == True:
    #update start position
    startpos = startpos+movesWnr
    #recursion
    n_valid, illegal_move, type_illegal = validation(n_valid, model, tokenizer, startpos, board, n_tokens)
  return n_valid, illegal_move, type_illegal

def test_game(model, tokenizer, startpos, n_tokens = 30):
  """
    Testing the model on a specific input
       - model = model that gets tested
       - tokenizer = tokenizer used for encoding and decoding
       - startpos = single input sequence
       - n_tokens = amount of tokens that should be generated
  """
  n_valid = 0
  board = chess.Board()
  #prepare start position sequence
  startpos_with_nr, startpos = prepare_startpos(startpos)
  #generate start position
  for move in startpos:
    board.push_san(move)
  #check generated game
  n_valid, illegal_move, type_illegal = validation(n_valid, model, tokenizer, startpos_with_nr, board, n_tokens)
  return n_valid, illegal_move, type_illegal

def test(model, tokenizer, startpos, n_tokens=30):
  """
    Testing the model on all start positions of a evaluation metric
       - model = model that gets tested
       - tokenizer = tokenizer used for encoding and decoding
       - startpos = (multiple) input sequences
       - n_tokens = amount of tokens that should be generated
  """
  global_n_valid = []
  global_illegal_move = []
  global_type_illegal = []
  #test model on each startposition
  for pos in startpos:
    n_valid, illegal_move, type_illegal = test_game(model, tokenizer, pos, n_tokens)
    global_n_valid.append(n_valid)
    global_illegal_move.append(illegal_move)
    global_type_illegal.append(type_illegal)
  #calculate average # of valid moves
  avg_valid = sum(global_n_valid) / len(global_n_valid)
  return avg_valid, global_n_valid, global_illegal_move, global_type_illegal

def eval(train_ds, test_ds, model, tokenizer, startpos, n_tokens=30):
  """
    Evaluation of one model at a specific training time
      - train_ds = dataset containing training data
      - test_ds = dataset containing test data
      - model = model that was saved during training
      - tokenizer = tokenizer used for encoding and decoding
      - startpos = list of all start positions
      - n_tokens = amount of tokens that should be generated
  """
  #test the model on all startpositions
  print("Start test for eval 1")
  eval1, global_n_valid1, global_illegal_move1, global_type_illegal1 = test(model, tokenizer, startpos[0], n_tokens)
  print("Start test for eval 2")
  eval2, global_n_valid2, global_illegal_move2, global_type_illegal2 = test(model, tokenizer, startpos[1], n_tokens)
  print("Start test for eval 3")
  eval3, global_n_valid3, global_illegal_move3, global_type_illegal3 = test(model, tokenizer, startpos[2], n_tokens)
  print("Start test for eval 4")
  eval4, global_n_valid4, global_illegal_move4, global_type_illegal4 = test(model, tokenizer, startpos[3], n_tokens)
  print("Start test for eval 5")
  eval5, global_n_valid5, global_illegal_move5, global_type_illegal5 = test(model, tokenizer, startpos[4], n_tokens)
  #combine data for simplicity
  evals = [eval1, eval2, eval3, eval4, eval5]                                                                                          #contains average # of valid moves for all evaluation metrics
  global_n_valid = [global_n_valid1, global_n_valid2, global_n_valid3, global_n_valid4, global_n_valid5]                               #contains # of valid moves of each generated game
  global_illegal_move = [global_illegal_move1, global_illegal_move2, global_illegal_move3, global_illegal_move4, global_illegal_move5] #contains illegal move of each generated game
  global_type_illegal = [global_type_illegal1, global_type_illegal2, global_type_illegal3, global_type_illegal4, global_type_illegal5] #contains the type of illegal move of each generated game
  return evals, global_n_valid, global_illegal_move, global_type_illegal

def global_eval(train_ds, test_ds, models, tokenizer, steps, n_tokens=30):
  """
    Evaluation of all models
      - train_ds = dataset containing training data
      - test_ds = dataset containing test data
      - models = list of all models that were saved during training
      - tokenizer = tokenizer used for encoding and decoding
      - steps = list of # of training steps each model was trained on
      - n_tokens = amount of tokens that should be generated
  """
  m_evals = []
  m_global_n_valid = []
  m_global_illegal_move = []
  m_global_type_illegal = []
  #create startpositions for model evaluation
  eval1_startpos = generate_common_openings(num_games=5, training_data=train_ds)
  eval2_startpos = generate_startpos(num_games=5, num_moves=5, testing_data=test_ds)
  eval3_startpos = generate_startpos(num_games=5, num_moves=10, testing_data=test_ds)
  eval4_startpos = generate_random_startpos(num_games=5, num_moves=5)
  eval5_startpos = generate_random_startpos(num_games=5, num_moves=10)
  #combine all start positions for simplicity
  startpos = [eval1_startpos, eval2_startpos, eval3_startpos, eval4_startpos, eval5_startpos]
  print("All startpositons: [eval1,eval2,eval3,eval4,eval5]")
  print(startpos)
  print()
  #evaluate all models
  for model in models:
    print("Evaluate new model"+"\n")
    evals, global_n_valid, global_illegal_move, global_type_illegal = eval(train_ds, test_ds, model, tokenizer, startpos, n_tokens)
    #save data
    m_evals.append(evals)
    m_global_n_valid.append(global_n_valid)
    m_global_illegal_move.append(global_illegal_move)
    m_global_type_illegal.append(global_type_illegal)
    print("Done evaluating current model"+"\n"+"--------------------------------"+"\n")
  print("Done with evaluating models"+"\n")
  #visualization
  create_graph(steps, m_evals)
  create_table(m_global_n_valid, m_global_illegal_move, m_global_type_illegal)
