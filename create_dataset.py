# Create simple dataset from games using huggingface tokenizer
# games.txt will include a list of 10 games in PGN format

from read_games import read_games
from transformers import TextDataset

#games = read_games("../lichess_db_2.pgn", 10, True, "games.txt") # read 10 games

def create_dataset(file_path, tokenizer, block_size=96): # create dataset from games
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
        overwrite_cache = True
    )
    return dataset
