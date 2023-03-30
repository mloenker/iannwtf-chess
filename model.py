from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments


class Model:
    def __init__(self, model_name):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.data_collator = self.load_data_collator(self.tokenizer)

    def load_data_collator(self, tokenizer):
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )

    def train(self, train_dataset, eval_dataset, output_dir, per_device_train_batch_size, num_train_epochs, save_steps, eval_steps, warmup_steps=0):
        '''
        POTENTIALLY NEEDS TO BE UPDATED TO INCLUDE A VALIDATION DATASET
        save_steps: number of steps after which a copy of the model is saved, might be useful for evaluation
        '''
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            warmup_steps=warmup_steps,
            save_strategy="steps",
            evaluation_strategy="steps",
            logging_strategy="steps",
            save_steps=save_steps,
            eval_steps=eval_steps
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
            
        trainer.train()
        trainer.save_model()

    def load(self, model_dir):
        '''
        Load a trained model from directory
        '''
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    def generate(self, prompt, max_new_tokens, num_return_sequences):
        '''
        prompt: string in LAN or PGN notation
        max_new_tokens: maximum number of tokens to generate (for PGN min. 6, for LAN 1). Set this to a high value to generate a full game or longer move sequences
        num_return_sequences: number of possible sequences of moves to return
        '''
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        beam_output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_beams=5,
            num_return_sequences=num_return_sequences,
            temperature=1.0,
            pad_token_id=self.model.config.eos_token_id,
            top_k=50,
            top_p=0.95
        )
        return [self.tokenizer.decode(beam, skip_special_tokens=True, clean_up_tokenization_spaces=True) for beam in beam_output]

    
    def generate_legal_move(self, prompt, board):
        '''
        Uses constrained beam search and board evaluation to generate a single legal move.
        Can only output one move because of the legality check, use generate() for multiple moves.
        This is deterministic and does not allow for sampling.
        '''
        legal_moves = [board.uci(move) for move in list(board.legal_moves)]
        force_words_ids = [self.tokenizer(legal_moves, add_prefix_space=True, add_special_tokens=False).input_ids]
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        beam_output = self.model.generate(
            input_ids,
            force_words_ids=force_words_ids,
            max_new_tokens=1,
            num_beams=10,
            num_return_sequences=1,
            temperature=1.0
        )
        return self.tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    

    def add_vocab(self, vocab_file):
        '''
        Add missing tokens to the vocabulary (used for LAN tokens)
        '''
        with open(vocab_file, "r") as file:
            vocab = file.read().splitlines()
        print("Tokens to be added: ", len(set(vocab) - set(self.tokenizer.get_vocab())))
        self.tokenizer.add_tokens(list(set(vocab) - set(self.tokenizer.get_vocab())))
        self.model.resize_token_embeddings(len(self.tokenizer))
        print("New vocabulary size: ", len(self.tokenizer))
    