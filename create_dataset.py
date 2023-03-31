# Create simple dataset from games using huggingface tokenizer
# games.txt will include a list of X games in PGN format

from transformers import TextDataset

def create_dataset(file_path, tokenizer, block_size=96): # create dataset from games
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
        overwrite_cache = True
    )
    return dataset
