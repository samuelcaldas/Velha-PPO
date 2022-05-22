import Velha


def main():
    velha = Velha.Velha()
    while True:
        velha.desenhaTabuleiro()
        print("Jogador %s" % velha.jogador)
        x = int(input("Digite a linha: "))
        y = int(input("Digite a coluna: "))
        winner = velha.jogar(x, y)
        print(winner)
        if (winner != 0):
            if (winner == 1):
                print("Jogador 1 venceu!")
            elif (winner == 2):
                print("Jogador 2 venceu!")
            elif (winner == -1):
                print("O tabuleiro está cheio!")
            elif (winner == -2):
                print("Posição inválida!")
            else:
                print("Não houve vencedor!")
            print("Score: %s" % velha.score)
            print("")
            opcao = input("Novo jogo? (s/n) ")
            if (opcao == "s"):
                velha.novoJogo()
            else:
                break


if __name__ == "__main__":
    main()
