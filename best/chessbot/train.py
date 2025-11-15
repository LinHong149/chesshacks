import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ChessDataset
from .model import ChessPolicyNet
from .config import BATCH_SIZE, LR, EPOCHS, MOVE_INDEX_PATH


def train():
    with open(MOVE_INDEX_PATH) as f:
        move_index_map = json.load(f)
    num_moves = len(move_index_map)

    dataset = ChessDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ChessPolicyNet(num_moves).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for boards, moves in tqdm(loader):
            boards = boards.cuda()
            moves = moves.cuda()

            logits = model(boards)
            loss = loss_fn(logits, moves)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}: loss={total_loss:.4f}")

    torch.save(model.state_dict(), "chessbot_policy.pth")
    print("Model saved to chessbot_policy.pth")


if __name__ == "__main__":
    train()