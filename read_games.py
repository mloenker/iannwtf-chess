def read_games(raw_file_path, number_of_games, write_to_file=True, target_file_path="games.txt"):
    with open(raw_file_path, 'r') as file:
        games = []
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
    if write_to_file:
        with open(target_file_path, 'w') as file:
            file.write(str(games))
    else:
        return games
