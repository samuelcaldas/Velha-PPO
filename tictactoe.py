import numpy as np


class make:
    def __init__(self) -> None:
        self.observation_space = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int)
        self.player = 1
        self.score = [0, 0]
        self.action_space = np.array([0, 0], dtype=int)

    def render(self):
        print(" ")
        print(" %s | %s | %s " %
              (self.observation_space[0][0], self.observation_space[0][1], self.observation_space[0][2]))
        print("---|---|---")
        print(" %s | %s | %s " %
              (self.observation_space[1][0], self.observation_space[1][1], self.observation_space[1][2]))
        print("---|---|---")
        print(" %s | %s | %s " %
              (self.observation_space[2][0], self.observation_space[2][1], self.observation_space[2][2]))
        print(" ")

    def verify_board(self):
        for i in range(3):
            # Verifica linhas
            if self.observation_space[i][0] == self.observation_space[i][1] == self.observation_space[i][2] != 0:
                return self.observation_space[i][0]

            # Verifica colunas
            if self.observation_space[0][i] == self.observation_space[1][i] == self.observation_space[2][i] != 0:
                return self.observation_space[0][i]

        # Verifica diagonais
        if self.observation_space[0][0] == self.observation_space[1][1] == self.observation_space[2][2] != 0:
            return self.observation_space[0][0]

        if self.observation_space[0][2] == self.observation_space[1][1] == self.observation_space[2][0] != 0:
            return self.observation_space[0][2]

        # Verifica se o tabuleiro está cheio
        count = 0
        for i in range(3):
            for j in range(3):
                if self.observation_space[i][j] == 0:
                    count += 1

        if count == 0:
            return -1

        return 0

    def clean_board(self):
        self.observation_space = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    def store_move(self, x, y):
        if(self.observation_space[x][y] == 0):
            self.observation_space[x][y] = self.player
            return True
        else:
            return False

    def swap_player(self):
        if(self.player == 1):
            self.player = 2
        else:
            self.player = 1

    def clean_player(self):
        self.player = 1

    def increment_score(self, jogador):
        self.score[self.player-1] = +1

    def clean_score(self):
        self.score = [0, 0]

    def make_move(self, x, y):
        if (self.store_move(x, y)):
            winner = self.verify_board()
            if (winner > 0):
                self.increment_score(winner)
                self.clean_board()
                self.clean_player()
                return winner
            elif (winner == 0):
                self.swap_player()
                return 0
            elif (winner == -1):
                self.increment_score(winner)
                self.clean_board()
                self.clean_player()
                return -1
        else:
            return -2

    def new_game(self):
        self.clean_board()
        self.clean_player()

    def restart(self):
        self.clean_board()
        self.clean_player()
        self.clean_score()

    # Take an action and return the next state, reward, and done flag
    def step(self, action):
        self.action_space = action
        x = self.action_space[0]
        y = self.action_space[1]
        winner = self.make_move(x, y)
        if (winner == 0):
            return self.observation_space, 0, False, None
        elif (winner == -1):
            return self.observation_space, -1, True, None
        else:
            return self.observation_space, 1, True, winner

    def reset(self):
        self.new_game()
        return self.observation_space
