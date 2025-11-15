import math
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

# ==========================
# CONFIG
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOARD_SIZE = 8
# We'll encode moves as from_square * 64 + to_square: 0..4095 (no special promotion handling)
ACTION_SIZE = 64 * 64

# Network / training hyperparams
CHANNELS = 64
NUM_RES_BLOCKS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100_000

NUM_SELFPLAY_GAMES_PER_ITER = 10
NUM_TRAIN_STEPS_PER_ITER = 100
MCTS_SIMULATIONS = 200
CPUCT = 1.5

# ==========================
# BOARD ENCODING
# ==========================

def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Encode a python-chess Board into a tensor of shape (C, 8, 8).
    Planes:
      0-5: white pawn, knight, bishop, rook, queen, king
      6-11: black pawn, knight, bishop, rook, queen, king
      12: side to move (all ones if white, zeros if black)
    """
    piece_map = board.piece_map()
    planes = torch.zeros(13, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)

    piece_to_plane = {
        (chess.PAWN, True): 0,
        (chess.KNIGHT, True): 1,
        (chess.BISHOP, True): 2,
        (chess.ROOK, True): 3,
        (chess.QUEEN, True): 4,
        (chess.KING, True): 5,
        (chess.PAWN, False): 6,
        (chess.KNIGHT, False): 7,
        (chess.BISHOP, False): 8,
        (chess.ROOK, False): 9,
        (chess.QUEEN, False): 10,
        (chess.KING, False): 11,
    }

    for square, piece in piece_map.items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        plane_idx = piece_to_plane[(piece.piece_type, piece.color)]
        planes[plane_idx, row, col] = 1.0

    # side to move plane
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0
    else:
        planes[12, :, :] = 0.0

    return planes


def move_to_action_index(move: chess.Move) -> int:
    """
    Encode move as from * 64 + to
    (ignores promotion type; they will share the same index)
    """
    return move.from_square * 64 + move.to_square


def action_index_to_move(board: chess.Board, idx: int) -> chess.Move | None:
    """
    Decode index back to a move, if legal, else None.
    This is mainly useful if you want to sample from policy directly.
    In MCTS we mostly go move -> index, not index -> move.
    """
    from_sq = idx // 64
    to_sq = idx % 64
    move = chess.Move(from_sq, to_sq)
    if move in board.legal_moves:
        return move
    return None


def legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """
    Returns a mask of size ACTION_SIZE, 1 for legal moves, 0 otherwise.
    """
    mask = torch.zeros(ACTION_SIZE, dtype=torch.float32)
    for move in board.legal_moves:
        idx = move_to_action_index(move)
        mask[idx] = 1.0
    return mask

# ==========================
# NETWORK
# ==========================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(self, in_channels=13, channels=64, num_res_blocks=4, action_size=ACTION_SIZE):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, action_size)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, C, 8, 8)
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # logits, shape (B, ACTION_SIZE)

        # Value
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # (B, 1)

        return p, v.squeeze(-1)  # policy_logits, value_scalar


# ==========================
# MCTS
# ==========================

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None):
        self.board = board
        self.parent = parent
        self.children: dict[int, "MCTSNode"] = {}

        self.P = {}  # prior prob for each action index
        self.N = {}  # visit count
        self.W = {}  # total value
        self.Q = {}  # mean value

        self.is_expanded = False

    def expand(self, policy_probs: torch.Tensor):
        """
        policy_probs: tensor of shape (ACTION_SIZE,)
        """
        self.is_expanded = True
        legal_mask = legal_moves_mask(self.board)
        # mask illegal moves
        masked_policy = policy_probs * legal_mask
        if masked_policy.sum().item() > 0:
            masked_policy /= masked_policy.sum()
        else:
            # If something goes weird, fall back to uniform over legal moves
            masked_policy = legal_mask / legal_mask.sum()

        for move in self.board.legal_moves:
            a = move_to_action_index(move)
            self.P[a] = masked_policy[a].item()
            self.N[a] = 0
            self.W[a] = 0.0
            self.Q[a] = 0.0

    def is_leaf(self):
        return not self.is_expanded

    def best_child(self, c_puct=CPUCT):
        """
        Select action using PUCT: argmax_a (Q + U)
        """
        best_score = -float("inf")
        best_action = None
        total_N = sum(self.N.values()) + 1

        for a in self.P.keys():
            Q_a = self.Q[a]
            U_a = c_puct * self.P[a] * math.sqrt(total_N) / (1 + self.N[a])
            score = Q_a + U_a
            if score > best_score:
                best_score = score
                best_action = a

        return best_action

    def backup(self, value: float):
        """
        Backpropagate value up the tree.
        """
        # value is from perspective of current player at this node
        node = self
        while node.parent is not None:
            parent = node.parent
            # action that led from parent to node
            a = node._action_from_parent
            parent.N[a] += 1
            parent.W[a] += value
            parent.Q[a] = parent.W[a] / parent.N[a]

            # switch perspective for parent
            value = -value
            node = parent


def mcts_search(root: MCTSNode, net: AlphaZeroNet, num_simulations=MCTS_SIMULATIONS):
    """
    Run MCTS starting from root node.
    Afterward, use visit counts as policy target.
    """
    net.eval()

    for _ in range(num_simulations):
        node = root
        board = root.board

        # 1. SELECTION
        while not node.is_leaf() and not board.is_game_over():
            a = node.best_child()
            move = action_index_to_move(board, a)
            if move is None:
                # Shouldn't happen because we only store legal moves
                break
            board = board.copy()
            board.push(move)

            if a not in node.children:
                node.children[a] = MCTSNode(board, parent=node)
                node.children[a]._action_from_parent = a  # store which action led here

            node = node.children[a]

        # 2. EVALUATION / EXPANSION
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                value = 1.0
            elif result == "0-1":
                value = -1.0
            else:
                value = 0.0
        else:
            # convert board to tensor and evaluate with network
            state_tensor = encode_board(board).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                policy_logits, value = net(state_tensor)
                policy_probs = F.softmax(policy_logits[0], dim=0).cpu()

            node.expand(policy_probs)
            value = value.item()

        # 3. BACKUP
        node.backup(value)

    # Build policy target as visit counts
    visit_counts = torch.zeros(ACTION_SIZE, dtype=torch.float32)
    for a, n in root.N.items():
        visit_counts[a] = n
    if visit_counts.sum() > 0:
        visit_counts /= visit_counts.sum()

    return visit_counts


# ==========================
# SELF-PLAY & REPLAY BUFFER
# ==========================

Sample = namedtuple("Sample", ["state", "policy", "value"])

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value):
        self.buffer.append(Sample(state, policy, value))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([b.state for b in batch])
        policies = torch.stack([b.policy for b in batch])
        values = torch.tensor([b.value for b in batch], dtype=torch.float32)
        return states, policies, values

    def __len__(self):
        return len(self.buffer)


def self_play_game(net: AlphaZeroNet, num_simulations=MCTS_SIMULATIONS):
    """
    Play one game using MCTS-guided self-play.
    Returns list of (state_tensor, policy_tensor, result_from_current_player).
    """
    net.eval()
    board = chess.Board()
    samples = []

    # Track players: +1 for white, -1 for black
    while not board.is_game_over():
        # encode current state
        state = encode_board(board)

        # create root and run MCTS
        root = MCTSNode(board.copy())
        policy_target = mcts_search(root, net, num_simulations=num_simulations)

        # temperature scheduling: at the start, sample; later, argmax
        move_probs = policy_target.clone()
        legal_mask = legal_moves_mask(board)
        move_probs *= legal_mask
        if move_probs.sum() > 0:
            move_probs /= move_probs.sum()
        else:
            # fallback uniform
            move_probs = legal_mask / legal_mask.sum()

        # pick move according to visit distribution
        action_idx = torch.multinomial(move_probs, 1).item()
        move = action_index_to_move(board, action_idx)
        if move is None:
            # emergency: random legal move
            move = random.choice(list(board.legal_moves))
            action_idx = move_to_action_index(move)

        samples.append((state, policy_target.clone(), board.turn))

        board.push(move)

    # Game over: assign result
    result = board.result()
    if result == "1-0":
        game_value_white = 1.0
    elif result == "0-1":
        game_value_white = -1.0
    else:
        game_value_white = 0.0

    # Convert to per-position value from that position's player perspective
    processed_samples = []
    for state, policy, turn in samples:
        # if turn == white, value = game_value_white
        # if turn == black, value = -game_value_white
        value = game_value_white if turn == chess.WHITE else -game_value_white
        processed_samples.append(
            Sample(state=state, policy=policy, value=value)
        )

    return processed_samples


# ==========================
# TRAINING
# ==========================

def train_step(net: AlphaZeroNet, optimizer, replay_buffer: ReplayBuffer):
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0, 0.0, 0.0

    states, policies, values = replay_buffer.sample(BATCH_SIZE)
    states = states.to(DEVICE)
    policies = policies.to(DEVICE)
    values = values.to(DEVICE)

    net.train()
    optimizer.zero_grad()

    policy_logits, value_pred = net(states)

    # Policy loss: cross entropy with target distribution
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policies * log_probs).sum(dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(value_pred, values)

    # L2 regularization via optimizer weight decay already
    loss = policy_loss + value_loss

    loss.backward()
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


def main():
    net = AlphaZeroNet(in_channels=13, channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    for iter_idx in range(1, 1000000):  # or some large number
        print(f"\n=== Iteration {iter_idx} ===")

        # 1. Self-play
        for g in range(NUM_SELFPLAY_GAMES_PER_ITER):
            print(f"  Self-play game {g+1}/{NUM_SELFPLAY_GAMES_PER_ITER}")
            samples = self_play_game(net, num_simulations=MCTS_SIMULATIONS)
            for s in samples:
                replay_buffer.add(s.state, s.policy, s.value)

        # 2. Training
        avg_loss = 0.0
        for step in range(NUM_TRAIN_STEPS_PER_ITER):
            loss, pl, vl = train_step(net, optimizer, replay_buffer)
            avg_loss += loss
        avg_loss /= max(1, NUM_TRAIN_STEPS_PER_ITER)
        print(f"  Avg training loss: {avg_loss:.4f} | Replay size: {len(replay_buffer)}")

        # 3. (Optional) Save checkpoint
        if iter_idx % 10 == 0:
            torch.save(net.state_dict(), f"alphazero_chess_iter_{iter_idx}.pt")
            print(f"  Saved model at iteration {iter_idx}")


if __name__ == "__main__":
    main()