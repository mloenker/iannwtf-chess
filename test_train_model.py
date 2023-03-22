from read_games import *
from create_dataset import *
from model import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel

'''
    # Train a first model on 10.000 games with 3 epochs 
'''

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

games = read_games("../lichess_db_2.pgn", 10000, True)
print("Read games")

dataset = create_dataset("games.txt", tokenizer, 96)
print("Created dataset")

collator = load_data_collator(tokenizer)

model = GPT2LMHeadModel.from_pretrained('gpt2')

print("Training model...")
train(model = model,
      tokenizer = tokenizer, 
      train_dataset = dataset, 
      data_collator = collator, 
      output_dir = "model_100k", 
      per_device_train_batch_size = 32, 
      num_train_epochs = 3,
      save_steps = 1000)