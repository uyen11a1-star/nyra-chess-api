
import os
import time
import threading
import tempfile
from pathlib import Path
from typing import Any

import requests
import chess
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# NYRA CHESS API
# CPU inference + Hugging Face private checkpoint
# ============================================================

SERVICE_NAME = "Nyra Chess API"
MODEL_URL = os.getenv("MODEL_URL", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "/tmp/nyra_chess_best.pt")
)

DEVICE = torch.device("cpu")
torch.set_num_threads(1)

app = FastAPI(title=SERVICE_NAME, version="1.0.0")

model = None
model_loaded = False
model_error = None
model_status = "loading"
model_lock = threading.Lock()
load_started = False


# ============================================================
# MODEL ARCHITECTURE
# Matches checkpoint keys: blocks.*.net.* and policy.*
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels, channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class NyraNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                18, 64, kernel_size=3,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(4)
        ])

        self.policy = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128, 4096),
        )

    def forward(self, x):
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        return self.policy(x)


# ============================================================
# CHECKPOINT DOWNLOAD
# ============================================================

def download_checkpoint():
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
        print(f"Using existing checkpoint: {MODEL_PATH}")
        return

    if not MODEL_URL:
        raise RuntimeError(
            "MODEL_URL is missing. Set it in Render Environment."
        )

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = MODEL_PATH.with_suffix(".download")

    try:
        print("Downloading Nyra Chess checkpoint...")

        with requests.get(
            MODEL_URL,
            headers=headers,
            stream=True,
            timeout=(30, 300),
            allow_redirects=True,
        ) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as output:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        output.write(chunk)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError("Downloaded checkpoint is empty.")

        temp_path.replace(MODEL_PATH)
        print(
            f"Checkpoint downloaded: "
            f"{MODEL_PATH.stat().st_size} bytes"
        )

    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint format is not a dictionary.")

    for key in ("model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate

    # A raw state_dict contains parameter tensors directly.
    if checkpoint and all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise RuntimeError(
        "Cannot find model_state_dict or state_dict in checkpoint."
    )


def load_model():
    global model, model_loaded, model_error, model_status

    with model_lock:
        model_status = "loading"
        model_loaded = False
        model_error = None

    try:
        download_checkpoint()

        print("Reading checkpoint on CPU...")
        try:
            checkpoint = torch.load(
                MODEL_PATH,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            # Compatibility fallback for older PyTorch versions.
            checkpoint = torch.load(
                MODEL_PATH,
                map_location="cpu",
            )

        state_dict = extract_state_dict(checkpoint)

        # Handle checkpoints saved through DataParallel.
        if state_dict and all(
            key.startswith("module.")
            for key in state_dict
        ):
            state_dict = {
                key[len("module."):]: value
                for key, value in state_dict.items()
            }

        candidate = NyraNet().to(DEVICE)

        # Strict loading: do not silently ignore missing weights.
        candidate.load_state_dict(state_dict, strict=True)
        candidate.eval()

        # Confirm the policy head produces 4096 move classes.
        with torch.inference_mode():
            test_input = torch.zeros(1, 18, 8, 8, device=DEVICE)
            test_output = candidate(test_input)

        if tuple(test_output.shape) != (1, 4096):
            raise RuntimeError(
                f"Unexpected model output shape: "
                f"{tuple(test_output.shape)}"
            )

        with model_lock:
            model = candidate
            model_loaded = True
            model_status = "ready"
            model_error = None

        print("Nyra Chess model loaded successfully.")

    except Exception as exc:
        with model_lock:
            model = None
            model_loaded = False
            model_status = "error"
            model_error = f"{type(exc).__name__}: {exc}"

        print(f"MODEL LOAD FAILED: {model_error}")


def start_model_loading():
    global load_started

    with model_lock:
        if load_started:
            return
        load_started = True

    thread = threading.Thread(
        target=load_model,
        daemon=True,
        name="nyra-model-loader",
    )
    thread.start()


@app.on_event("startup")
def startup_event():
    start_model_loading()


# ============================================================
# BOARD ENCODING: 18 PLANES
# 0-5 white pieces, 6-11 black pieces
# 12 side to move
# 13-16 castling rights
# 17 en-passant square
# ============================================================

PIECE_TO_PLANE = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}


def encode_board(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros((18, 8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.color, piece.piece_type)]
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        planes[plane, rank, file] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        planes[17, rank, file] = 1.0

    return planes.unsqueeze(0)


# ============================================================
# API SCHEMAS
# ============================================================

class MoveRequest(BaseModel):
    fen: str
    legal_moves: list[Any] = []


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "service": SERVICE_NAME,
        "health": "/health",
        "move": "/move",
    }


@app.get("/health")
def health():
    with model_lock:
        return {
            "status": model_status,
            "model_loaded": model_loaded,
            "model_error": model_error,
        }


@app.post("/move")
def choose_move(request: MoveRequest):
    with model_lock:
        current_model = model
        ready = model_loaded

    if not ready or current_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model is not ready",
                "model_status": model_status,
                "model_error": model_error,
            },
        )

    try:
        board = chess.Board(request.fen)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid FEN: {exc}",
        )

    if not board.is_valid():
        raise HTTPException(
            status_code=400,
            detail="Invalid chess position.",
        )

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        raise HTTPException(
            status_code=400,
            detail="No legal moves in this position.",
        )

    # Use legal moves supplied by the frontend when available.
    # Every move is validated against the actual board.
    if request.legal_moves:
        allowed_uci = set()

        for item in request.legal_moves:
            if isinstance(item, str):
                raw_uci = item
            elif isinstance(item, dict):
                raw_uci = item.get("uci")
            else:
                continue

            if not raw_uci:
                continue

            try:
                candidate_move = chess.Move.from_uci(raw_uci)
                if candidate_move in board.legal_moves:
                    allowed_uci.add(candidate_move.uci())
            except (ValueError, TypeError):
                continue

        legal_moves = [
            move for move in legal_moves
            if move.uci() in allowed_uci
        ]

        if not legal_moves:
            raise HTTPException(
                status_code=400,
                detail="No valid legal_moves were supplied.",
            )

    try:
        input_tensor = encode_board(board).to(DEVICE)

        with torch.inference_mode():
            logits = current_model(input_tensor)[0]

        # Map each move to from_square * 64 + to_square.
        # Promotions share the same policy index, so prefer queen
        # promotion when multiple promotions have the same score.
        ranked_moves = []

        for move in legal_moves:
            move_index = move.from_square * 64 + move.to_square
            score = float(logits[move_index].item())

            promotion_bonus = (
                1e-6 if move.promotion == chess.QUEEN else 0.0
            )

            ranked_moves.append(
                (score + promotion_bonus, move)
            )

        ranked_moves.sort(key=lambda item: item[0], reverse=True)
        best_move = ranked_moves[0][1]

        return {
            "uci": best_move.uci(),
            "san": board.san(best_move),
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"MOVE ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Move inference failed: {type(exc).__name__}: {exc}",
        )
