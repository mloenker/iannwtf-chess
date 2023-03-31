## Overview
We finetuned two GPT2-small models using PGN data from lichess.org. Both models were trained on 100k games.

The [SAN Model](https://www.mediafire.com/file/rubrdho1go52rde/model_san.zip/file) uses standard algebraic notation and makes illegal moves around move 10.

The [LAN Model](http://www.mediafire.com/file/o4tfjk3rbi954d2/model_lan.zip) uses long algebraic notation and will always find a legal move using constrained beam search.

## Installation requirements 
```
pip install chess
pip install transformers
pip install tabulate
```

## Document overview
| Document      | Content       |
| ------------- | ------------- |
| demo.ipynb | demo notebook; showcasing training and playing a game against the model |
| project_report.pdf | our project report including all sources |
| meeting_summary.pdf | summary of contents discussed in the meeting(s) with tutor |
| read_games.py | read either SAN or LAN notation from PGN files and store the games |
| create_dataset.py | create dataset from stored games |
| model.py | model class; includes functions for training and generating |
| game.py | game class; play games against the model |
| create_eval_startpos.py | all functions needed to create the start positions for evaluation |
| eval_funcs.py | all functions needed for evaluation |
| create_eval_visual.py | graph and table creation for evaluation |
| main_PGN_model.py | all code needed to train and evaluate the SAN model ****NEEDS CHANGE???****|
| lan_vocab.txt | added vocabulary for LAN model |

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
	human_start=True # who makes the first move; if you start from the base position the human must always make the first move
)
game.play()
```


