# Create simple dataset from games using huggingface tokenizer

from read_games import read_games
from transformers import GPT2Tokenizer, TextDataset
from sklearn.model_selection import train_test_split

games = read_games("../lichess_db_2.pgn", "games.txt" , 10) # read 10 games

tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # load tokenizer

def create_dataset(file_path, tokenizer): # create dataset from games
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size=96,
        overwrite_cache=True
    )
    return dataset
