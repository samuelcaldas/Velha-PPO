class make:
    def __init__(self) -> None:
        self.tabuleiro = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.player = 1
        self.score = [0, 0]
        self.observation_space = (3, 3)
        self.action_space = 2

    def render(self):
        print(" ")
        print(" %s | %s | %s " %
              (self.tabuleiro[0][0], self.tabuleiro[0][1], self.tabuleiro[0][2]))
        print("---|---|---")
        print(" %s | %s | %s " %
              (self.tabuleiro[1][0], self.tabuleiro[1][1], self.tabuleiro[1][2]))
        print("---|---|---")
        print(" %s | %s | %s " %
              (self.tabuleiro[2][0], self.tabuleiro[2][1], self.tabuleiro[2][2]))
        print(" ")

    def verificaTabuleiro(self):
        for i in range(3):
            # Verifica linhas
            if self.tabuleiro[i][0] == self.tabuleiro[i][1] == self.tabuleiro[i][2] != 0:
                return self.tabuleiro[i][0]

            # Verifica colunas
            if self.tabuleiro[0][i] == self.tabuleiro[1][i] == self.tabuleiro[2][i] != 0:
                return self.tabuleiro[0][i]

        # Verifica diagonais
        if self.tabuleiro[0][0] == self.tabuleiro[1][1] == self.tabuleiro[2][2] != 0:
            return self.tabuleiro[0][0]

        if self.tabuleiro[0][2] == self.tabuleiro[1][1] == self.tabuleiro[2][0] != 0:
            return self.tabuleiro[0][2]

        # Verifica se o tabuleiro está cheio
        count = 0
        for i in range(3):
            for j in range(3):
                if self.tabuleiro[i][j] == 0:
                    count += 1

        if count == 0:
            return -1

        return 0

    def limpaTabuleiro(self):
        self.tabuleiro = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    def gravaPosicao(self, x, y):
        if(self.tabuleiro[x][y] == 0):
            self.tabuleiro[x][y] = self.player
            return True
        else:
            return False

    def trocaJogador(self):
        if(self.player == 1):
            self.player = 2
        else:
            self.player = 1

    def limpaJogador(self):
        self.player = 1

    def incrementaScore(self, jogador):
        self.score[self.player-1] = +1

    def limpaScore(self):
        self.score = [0, 0]

    def jogar(self, x, y):
        if (self.gravaPosicao(x, y)):
            winner = self.verificaTabuleiro()
            if (winner > 0):
                self.incrementaScore(winner)
                self.limpaTabuleiro()
                self.limpaJogador()
                return winner
            elif (winner == 0):
                self.trocaJogador()
                return 0
            elif (winner == -1):
                self.incrementaScore(winner)
                self.limpaTabuleiro()
                self.limpaJogador()
                return -1
        else:
            return -2

    def novoJogo(self):
        self.limpaTabuleiro()
        self.limpaJogador()

    def restart(self):
        self.limpaTabuleiro()
        self.limpaJogador()
        self.limpaScore()

    # Take an action and return the next state, reward, and done flag
    def step(self, action):
        x = action[0]
        y = action[1]
        winner = self.jogar(x, y)
        if (winner == 0):
            return self.tabuleiro, 0, False, None
        elif (winner == -1):
            return self.tabuleiro, -1, True, None
        else:
            return self.tabuleiro, 1, True, winner

    def reset(self):
        self.novoJogo()
        return self.tabuleiro
