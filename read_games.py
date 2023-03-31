import re

def read_games(raw_file_path, number_of_games, format="pgn", write_to_file=True, target_file_path="games.txt", startpoint=0):
    number_of_games = number_of_games + startpoint
    with open(raw_file_path, 'r') as file:
        games = []
        if format == "pgn" or format == "san":
            for line in file:
                if line[0] == "[": continue # remove metadata
                elif line == '\n': continue # remove empty lines
                elif "eval" in line: continue # remove evals
                elif "#" not in line: continue # remove games that are not finished by checkmate
                elif "100." in line: continue # remove games that are too long (at least 100 moves)
                else:
                    games.append(line)
                    number_of_games-=1
                if number_of_games==0: break
        elif format == "lan":
            for line in file:
                if line[0] == "[": continue # remove metadata
                elif line == '\n': continue # remove empty lines
                else:
                    games.append(line)
                    number_of_games-=1
                if number_of_games==0: break
    
    games = games[startpoint:]

    if write_to_file:
        with open(target_file_path, 'w') as file:
            file.write(str(games))
    else:
        return games
