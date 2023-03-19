# Create simple dataset from games using huggingface tokenizer
# games.txt will include a list of 10 games in PGN format

from read_games import read_games
from transformers import GPT2Tokenizer, TextDataset

games = read_games("../lichess_db_2.pgn", 10, True, "games.txt") # read 10 games

tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # load tokenizer

def create_dataset(file_path, tokenizer): # create dataset from games
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size=96,
        overwrite_cache=True
    )
    return dataset
