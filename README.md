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
| read_games.py | read either SAN or LAN notation from PGN files and store the games |
| create_dataset.py | create dataset from stored games |
| model.py | model class; includes functions for training and generating |
| game.py | game class; play games against the model |
| create_eval_startpos.py | all functions needed to create the start positions for evaluation |
| create_eval_visual.py | graph creation for evaluation |
| lan_vocab.txt | added vocabulary for LAN model |

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

