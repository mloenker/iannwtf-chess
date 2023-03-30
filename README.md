## Installation requirements 
```
pip install chess
pip install transformers
pip install tabulate
```

## Document overview
| Document      | Content       |
| ------------- | ------------- |
| project_report.pdf | our project report including all sources |
| meeting_summary.pdf | summary of contents discussed in the meeting(s) with tutor |
| read_games.py | function to remove all unnecessary information from the game transcripts & combine data from multiple sources into one file |
| create_dataset.py | function to create the dataset from game transcripts |
| games.txt | example dataset |
| model.py | functions for training and testing the model |
| test_train_model.py | main code |
| test_game_stats.ipynb | analyse of game data |
| test_play_game.ipynb | Example output of trained model |
| create_eval_startpos | all functions needed to create the start positions for evaluation |

## Remaining things to discuss
*** code in test_train_model.py not necessary as separe file??? Included pure code in main_PGN_model.ipynb <br>
*** what is the functionality of Finetuning.ipynb? <br>
*** main_PGN_model.py was not tested in this environment yet (code - not imports - was copied from Jupyter Notebooks). It might be wise to switch to .ipynb format for the visualizations.

## Demo
Load the model
```python
from model import Model
from game import Game

model = Model("gpt2") # load the base gpt2 model
model.add_vocab("lan_vocab.txt") # add_vocab is required for LAN models
model.load("model_28") # load a finetuned model from files
```
Play a game against the model
```python
game = Game(model,
	model_format="lan", # model format; use either "san"/"pgn" or "lan"/"uci"
	human_start=True, # who makes the first move; if you start from the base position the human must always make the first move
	start_position="8/8/8/2k5/4K3/8/8/8 w - - 4 45" # leave empty if you want to start from the base position; format is FEN
)
game.play()
```
![output](https://user-images.githubusercontent.com/103146401/228558749-8e2f89e3-e08f-4bf7-a817-5287f5f89f58.svg)

