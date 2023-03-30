import chess
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import tensorflow as tf
from collections import Counter
from transformers import TextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from tabulate import tabulate
from read_games import read_games
from create_dataset import create_dataset
from model import Model
from create_eval_startpos import extract_start, generate_common_openings, generate_startpos, generate_random_state, generate_random_startpos
from create_eval_visual import create_graph, create_table
from eval_funcs import prepare_startpos, generate_game, validation, test_game, test, eval, global_eval

#test if gpu is enabled
tf.test.gpu_device_name()

# bash code to mount the drive
import os
from google.colab import drive
drive.mount ("/content/drive")
os.chdir("/content/drive/MyDrive")

#instantiate model
model = Model(model_name='gpt2')
print("Instantiated model")

#create datasets
games_train = read_games(raw_file_path="/content/drive/MyDrive/Chess/lichess_db_2015-08.pgn", number_of_games=10000, format="pgn", write_to_file=True, target_file_path="games_train.txt", startpoint=0)
games_eval = read_games(raw_file_path="/content/drive/MyDrive/Chess/lichess_db_2013-01.pgn", number_of_games=50, format="pgn", write_to_file=True, target_file_path="games_eval.txt", startpoint=0)
print("Read games")
train_ds = create_dataset(file_path="games_train.txt", tokenizer=model.tokenizer, block_size=96)
eval_ds = create_dataset(file_path="games_eval.txt", tokenizer=model.tokenizer, block_size=96)
print("Created datasets")

#train model
model.train(train_dataset=train_ds, eval_dataset=eval_ds, output_dir="model_100k", per_device_train_batch_size=32, num_train_epochs=5, save_steps=1000)
print("Model trained")

#define checkpoints (must have been saved during training)
checkpoints = [3000,4000]

#load saved models
models = []
for step in range(len(checkpoints)):
  model_dir = "./model_100k/checkpoint-" + str(checkpoints[step])
  model.load(model_dir)
  models.append(model)
print("Loaded saved models")
print(models)
print()

#evaluate models
train_ds = read_games("/content/drive/MyDrive/Chess/lichess_db_2015-08.pgn", 10000, write_to_file=False, target_file_path="games_train.txt")
print("Train_ds: ")
print(train_ds[:2])
test_ds = read_games("/content/drive/MyDrive/Chess/lichess_db_2013-01.pgn", 50, write_to_file=False, target_file_path="games_test.txt")
print("Test_ds: ")
print(test_ds[:2])
print()
global_eval(train_ds, test_ds, models, model.tokenizer, checkpoints)
