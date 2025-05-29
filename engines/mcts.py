import math, chess, torch
from collections import defaultdict
from auxiliary_func import board_to_matrix

class MCTS:
    def __init__(self, model, move_to_int, int_to_move, c_puct=1.0, device='cpu'):
        self.model = model
        self.move_to_int = move_to_int
        self.int_to_move = int_to_move
        self.c_puct = c_puct
        self.device = device

        self.Q = defaultdict(float)      # sum of values
        self.N = defaultdict(int)        # visit counts
        self.P = {}                      # priors for each state
        self.children = {}               # legal moves per state

    def run(self, root_board, simulations=200):
        for _ in range(simulations):
            self._simulate(root_board.copy())
        state = root_board.fen()
        moves = self.children[state]
        visits = [self.N[(state, mv)] for mv in moves]
        best = moves[visits.index(max(visits))]
        return best

    def _simulate(self, board):
        state = board.fen()
        # 1) expansion & evaluation if leaf node
        if state not in self.children:
            # inference
            X = torch.tensor(board_to_matrix(board),
                             dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, value = self.model(X)
                logits = logits.squeeze(0).cpu().numpy()
                value = value.item()
            # priors for legal moves
            legal = list(board.legal_moves)
            priors = {}
            for mv in legal:
                uci = mv.uci()
                if uci in self.move_to_int:
                    idx = self.move_to_int[uci]
                    priors[mv] = math.exp(logits[idx])
                else:
                    # unseen move: assign small prior
                    priors[mv] = 1e-8
            total_prior = sum(priors.values()) or 1.0
            for mv in priors:
                priors[mv] /= total_prior

            self.P[state] = priors
            # children list for this state
            self.children[state] = legal
            return value

        # 2) selection
        best_score, best_move = -float('inf'), None
        total_N = sum(self.N[(state, mv)] for mv in self.children[state])
        for mv in self.children[state]:
            q = self.Q[(state, mv)] / (1 + self.N[(state, mv)])
            u = self.c_puct * self.P[state][mv] * math.sqrt(total_N) / (1 + self.N[(state, mv)])
            if q + u > best_score:
                best_score, best_move = q + u, mv

        board.push(best_move)
        v = self._simulate(board)
        # backpropagate
        self.N[(state, best_move)] += 1
        self.Q[(state, best_move)] += v
        return v
