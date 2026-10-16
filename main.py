import os
import threading
from pathlib import Path

import requests
import torch
import torch.nn as nn
import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- Configuration ----------
MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/nyra_chess_best.pt")
MODEL_URL = os.getenv("MODEL_URL", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
DEVICE = torch.device("cpu")

app = FastAPI(title="Nyra Chess API", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Model architecture ----------
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)


class NyraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(64) for _ in range(4)))
        self.policy_conv = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
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
model_error = "Model has not loaded yet."
model_lock = threading.Lock()


# ---------- Checkpoint download and loading ----------
def download_checkpoint():
    path = Path(MODEL_PATH)
    if path.is_file() and path.stat().st_size > 0:
        return

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL is missing. Set it in Render Environment.")

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    temporary_path = path.with_suffix(path.suffix + ".download")
    with requests.get(
        MODEL_URL, headers=headers, timeout=(20, 300), stream=True
    ) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)

    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        raise RuntimeError("Downloaded checkpoint is empty.")

    temporary_path.replace(path)


def load_model():
    global model, model_error
    try:
        download_checkpoint()
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        if not isinstance(checkpoint, dict):
            raise RuntimeError("Checkpoint format is not a dictionary.")

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError("Could not find model_state_dict in checkpoint.")

        network = NyraNet()
        network.load_state_dict(state_dict, strict=True)
        network.to(DEVICE)
        network.eval()

        with model_lock:
            model = network
            model_error = None

        print("NYRA_MODEL_LOADED=true")

    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"
        print(f"NYRA_MODEL_LOAD_ERROR={model_error}")


# ---------- Board encoding ----------
def encode_board(board):
    # 18 planes: white pieces 0-5, black pieces 6-11, side-to-move 12,
    # castling rights 13-16, en-passant square 17.
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

    castling_rights = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for i, allowed in enumerate(castling_rights):
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

    # This model predicts from-square/to-square only, not promotion type.
    # Prefer queen promotion; if only underpromotions are legal, use one legal move.
    candidates = []
    for move in legal_moves:
        if move.promotion is not None and move.promotion != chess.QUEEN:
            continue
        class_index = move.from_square * 64 + move.to_square
        candidates.append((float(logits[class_index].item()), move))

    if not candidates:
        return legal_moves[0].uci()

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1].uci()


# Load asynchronously so the health endpoint can respond while loading.
threading.Thread(target=load_model, daemon=True).start()


class MoveRequest(BaseModel):
    fen: str
    legal_moves: list | None = None


@app.get("/")
def root():
    return {"service": "Nyra Chess API", "health": "/health", "move": "/move"}


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
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid FEN.")

    if not board.is_valid():
        raise HTTPException(status_code=400, detail="Invalid chess position.")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_error or "Nyra model is still loading.",
        )

    move_uci = choose_move(board)
    if move_uci is None:
        raise HTTPException(status_code=409, detail="This position has no legal moves.")

    return {"uci": move_uci}
