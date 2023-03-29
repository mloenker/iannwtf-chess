import chess
import chess.pgn

class Game:
    def __init__(self, model, model_format="lan", human_start=True, start_position=None):
        self.board = chess.Board()
        self.model = model
        self.model_format = model_format
        self.human_turn = human_start # True if human has first move, False if AI starts

        if model_format == "san" or "pgn": # python-chess requires an empty board to parse the move stack
            self.empty_board = chess.Board()

    def play(self):
        display(self.board) # Display the board position in a nice way
        print("Board position: ", self.get_board_position(self.model_format)) # Print board position in notation (this is what the model sees)

        try:
            move = self.get_move(self.human_turn)
        except:
            print("Invalid move text! Please use SAN or LAN notation.")
            
        if move in self.board.legal_moves:
            self.board.push(move) # MAKE THE MOVE (most important line of code)
            self.human_turn = not self.human_turn # Switch turns
        else:
            print("Invalid move! Please try again.")

        if not self.board.is_game_over(): # Check if the game is over, if not play the next round
            self.play()
        else:
            print("Game over!")

                
    def get_move(self, human_turn):
        # Get move from human
        if human_turn:
            move_text = str(input("Your move: "))
            move = self.board.parse_san(move_text)
            return move
        # Get move from AI
        else:
            board_position = self.get_board_position(self.model_format)
            if self.model_format == "pgn" or self.model_format == "san":
                moves = self.model.generate(board_position, 6, 5) # Generate top 5 strings of 6 tokens (the maximum length of a move in SAN notation)
                for move in moves:
                    move = move.replace(board_position+" ", "") # Keep only the newly generated string
                    move = move.split(" ")[0] # The first move in the string is the new move
                    move = self.board.parse_san(move)
                    if move in self.board.legal_moves:
                        return move
            elif self.model_format == "lan" or self.model_format == "uci":
                moves = self.model.generate(board_position, 1, 5) # Generate top 5 strings of 1 token (each move is one token)
                for move in moves:
                    move = move.split(" ")[-1] # Take the last token, which is the new move
                    move = self.board.parse_san(move)
                    if move in self.board.legal_moves:
                        return move
            # No legal move found
            if self.model_format == "lan" or self.model_format == "uci": # If model uses UCI/LAN format, we can force a legal move with constrained beam search
                move = self.model.generate_legal_move(board_position, self.board)
                move = self.board.parse_san(move)
                return move
            else: # If model uses PGN/SAN format, we cannot use constrained beam search, as moves consist of multiple tokens
                print("The model was not able to find a legal move. It will now resign.")
                return self.board.resign()
            
    def get_board_position(self, format):
        if format == "uci" or format == "lan": # Return the board position in LAN notation
            # Found no way to get LAN notation directly in documentation, so it has to be created manually
            text = ""
            for index, move in enumerate(self.board.move_stack):
                if index % 2 == 0:
                    text += str(int(index/2)+1) + ". "
                text += str(move) + " "
            return text
        elif format == "pgn" or format == "san": # Return the board position in SAN notation
            return self.empty_board.variation_san(self.board.move_stack)
        elif format == "fen": # Return the board position in FEN notation
            return self.board.fen()
        else:
            raise ValueError("Invalid format. Please use 'lan' or 'pgn'.")