import os
import threading
import requests
import torch
import torch.nn as nn
import chess

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

---------- Configuration ----------

MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/nyra_chess_best.pt")
MODEL_URL = os.getenv("MODEL_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

DEVICE = torch.device("cpu")
app = FastAPI(title="Nyra Chess API", version="1.0.0")

app.add_middleware(
CORSMiddleware,
allow_origins=[""],
allow_credentials=False,
allow_methods=[""],
allow_headers=["*"],
)

---------- Model architecture ----------

class ResidualBlock(nn.Module):
def init(self, channels=64):
super().init()
self.conv1 = nn.Conv2d(
channels, channels, 3, padding=1, bias=False
)
self.bn1 = nn.BatchNorm2d(channels)
self.conv2 = nn.Conv2d(
channels, channels, 3, padding=1, bias=False
)
self.bn2 = nn.BatchNorm2d(channels)

def forward(self, x):
    residual = x
    x = torch.relu(self.bn1(self.conv1(x)))
    x = self.bn2(self.conv2(x))
    return torch.relu(x + residual)

class NyraNet(nn.Module):
def init(self):
super().init()
self.stem = nn.Sequential(
nn.Conv2d(18, 64, 3, padding=1, bias=False),
nn.BatchNorm2d(64),
nn.ReLU(),
)
self.blocks = nn.Sequential(
*[ResidualBlock(64) for _ in range(4)]
)
self.policy_conv = nn.Sequential(
nn.Conv2d(64, 2, 1),
nn.ReLU(),
nn.Flatten(),
)
self.policy_fc = nn.Linear(128, 4096)

def forward(self, x):
    x = self.stem(x)
    x = self.blocks(x)
    x = self.policy_conv(x)
    return self.policy_fc(x)

model = None
model_error = None
model_lock = threading.Lock()

def download_model():
if os.path.isfile(MODEL_PATH):
return

if not MODEL_URL:
    raise RuntimeError(
        "Model checkpoint is missing. Configure MODEL_URL "
        "in Render environment variables."
    )

headers = {}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN}"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
response = requests.get(
    MODEL_URL, headers=headers, timeout=180
)
response.raise_for_status()

with open(MODEL_PATH, "wb") as file:
    file.write(response.content)

def load_model():
global model, model_error

try:
    download_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    network = NyraNet()
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(DEVICE)
    network.eval()

    with model_lock:
        model = network
        model_error = None

except Exception as exc:
    model_error = str(exc)
    print(f"NYRA MODEL LOAD ERROR: {exc}")

def encode_board(board):
x = torch.zeros((18, 8, 8), dtype=torch.float32)

piece_to_plane = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

for square, piece in board.piece_map().items():
    row = 7 - chess.square_rank(square)
    col = chess.square_file(square)
    plane = piece_to_plane[piece.piece_type]

    if piece.color == chess.BLACK:
        plane += 6

    x[plane, row, col] = 1.0

if board.turn == chess.WHITE:
    x[12].fill_(1.0)

castling = [
    board.has_kingside_castling_rights(chess.WHITE),
    board.has_queenside_castling_rights(chess.WHITE),
    board.has_kingside_castling_rights(chess.BLACK),
    board.has_queenside_castling_rights(chess.BLACK),
]

for i, allowed in enumerate(castling):
    if allowed:
        x[13 + i].fill_(1.0)

if board.ep_square is not None:
    row = 7 - chess.square_rank(board.ep_square)
    col = chess.square_file(board.ep_square)
    x[17, row, col] = 1.0

return x.unsqueeze(0)

def choose_move(board):
if model is None:
raise RuntimeError(model_error or "Model is not loaded.")

legal_moves = list(board.legal_moves)
if not legal_moves:
    return None

encoded = encode_board(board).to(DEVICE)

with torch.inference_mode():
    logits = model(encoded)[0]

# Rank legal moves only. Promotion moves use queen promotion
# because the trained 4096-class policy has no promotion class.
ranked = []

for move in legal_moves:
    if move.promotion is not None and move.promotion != chess.QUEEN:
        continue

    index = move.from_square * 64 + move.to_square
    ranked.append((float(logits[index].item()), move))

if not ranked:
    return chess.Move.from_uci(
        next(iter(legal_moves)).uci()
    ).uci()

ranked.sort(key=lambda item: item[0], reverse=True)
return ranked[0][1].uci()

Start loading in the background so health checks can respond.

threading.Thread(target=load_model, daemon=True).start()

class MoveRequest(BaseModel):
fen: str
legal_moves: list | None = None

@app.get("/")
def root():
return {"service": "Nyra Chess API"}

@app.get("/health")
def health():
return {
"status": "ok" if model is not None else "loading",
"model_loaded": model is not None,
"model_error": model_error,
}

@app.post("/move")
def get_move(request: MoveRequest):
try:
board = chess.Board(request.fen)
except ValueError:
raise HTTPException(status_code=400, detail="Invalid FEN")

if not board.is_valid():
    raise HTTPException(status_code=400, detail="Invalid chess position")

if model is None:
    raise HTTPException(
        status_code=503,
        detail=model_error or "Nyra model is still loading",
    )

move_uci = choose_move(board)

if move_uci is None:
    raise HTTPException(
        status_code=409,
        detail="The position has no legal moves",
    )

return {"uci": move_uci}
