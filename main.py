
import os
import time
import threading
from pathlib import Path
from typing import Any

import requests
import chess
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# NYRA CHESS API 0.2
# NyraNet + Negamax + Alpha-Beta
# CPU inference + private Hugging Face checkpoint
# ============================================================

SERVICE_NAME = "Nyra Chess API"
MODEL_URL = os.getenv("MODEL_URL", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/tmp/nyra_chess_best.pt"))

DEVICE = torch.device("cpu")
torch.set_num_threads(1)

SEARCH_DEPTH = max(1, min(4, int(os.getenv("SEARCH_DEPTH", "2"))))
MAX_SEARCH_NODES = max(
    50, min(5000, int(os.getenv("MAX_SEARCH_NODES", "250")))
)
SEARCH_TIME_LIMIT = max(
    0.25, min(15.0, float(os.getenv("SEARCH_TIME_LIMIT", "3.0")))
)

app = FastAPI(title=SERVICE_NAME, version="0.2.0")

# CORS: allow the GitHub Pages frontend to call Render.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://uyen11a1-star.github.io"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

model = None
model_loaded = False
model_error = None
model_status = "loading"
model_lock = threading.Lock()
load_started = False

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

MATE_SCORE = 100_000


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class NyraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(18, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(4)
        ])
        self.policy = nn.Sequential(
            nn.Conv2d(64, 2, 1),
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
# DOWNLOAD PRIVATE HUGGING FACE CHECKPOINT
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
            f"Checkpoint downloaded: {MODEL_PATH.stat().st_size} bytes"
        )

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


# ============================================================
# LOAD MODEL
# ============================================================

def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint format is not a dictionary.")

    for key in ("model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate

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
            checkpoint = torch.load(
                MODEL_PATH,
                map_location="cpu",
            )

        state_dict = extract_state_dict(checkpoint)

        if state_dict and all(
            key.startswith("module.") for key in state_dict
        ):
            state_dict = {
                key[len("module."):]: value
                for key, value in state_dict.items()
            }

        candidate = NyraNet().to(DEVICE)
        candidate.load_state_dict(state_dict, strict=True)
        candidate.eval()

        with torch.inference_mode():
            output = candidate(
                torch.zeros(1, 18, 8, 8, device=DEVICE)
            )

        if tuple(output.shape) != (1, 4096):
            raise RuntimeError(
                f"Unexpected model output shape: {tuple(output.shape)}"
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

    threading.Thread(
        target=load_model,
        daemon=True,
        name="nyra-model-loader",
    ).start()


@app.on_event("startup")
def startup_event():
    start_model_loading()


# ============================================================
# BOARD ENCODING: 18 PLANES
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
        planes[
            plane,
            chess.square_rank(square),
            chess.square_file(square),
        ] = 1.0

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
        planes[
            17,
            chess.square_rank(board.ep_square),
            chess.square_file(board.ep_square),
        ] = 1.0

    return planes.unsqueeze(0)


# ============================================================
# POSITION EVALUATION
# ============================================================

def evaluate_board(board: chess.Board) -> int:
    if board.is_checkmate():
        return -MATE_SCORE

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
        or board.is_repetition(3)
    ):
        return 0

    score = 0

    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)

        relative_rank = (
            rank_idx if piece.color == chess.WHITE else 7 - rank_idx
        )

        if piece.piece_type == chess.PAWN:
            value += relative_rank * 5

        elif piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            center_distance = (
                abs(3.5 - file_idx) + abs(3.5 - rank_idx)
            )
            value += int(12 - 3 * center_distance)

        score += value if piece.color == chess.WHITE else -value

    if board.is_check():
        score -= 20

    return score if board.turn == chess.WHITE else -score


# ============================================================
# NYRANET POLICY + MOVE ORDERING
# ============================================================

def get_policy_logits(board: chess.Board, current_model):
    x = encode_board(board).to(DEVICE)
    with torch.inference_mode():
        return current_model(x)[0]


def order_moves(board: chess.Board, moves, logits):
    ranked = []

    for move in moves:
        index = move.from_square * 64 + move.to_square
        score = float(logits[index].item())

        if move.promotion:
            score += 5000 + PIECE_VALUES.get(move.promotion, 0)

        if board.is_capture(move):
            victim = board.piece_at(move.to_square)

            if victim is None and board.is_en_passant(move):
                victim_value = PIECE_VALUES[chess.PAWN]
            else:
                victim_value = (
                    PIECE_VALUES[victim.piece_type] if victim else 0
                )

            attacker = board.piece_at(move.from_square)
            attacker_value = (
                PIECE_VALUES[attacker.piece_type] if attacker else 0
            )

            score += 10_000 + victim_value * 10 - attacker_value

        board.push(move)
        if board.is_check():
            score += 100
        board.pop()

        ranked.append((score, move))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [move for _, move in ranked]


# ============================================================
# NEGAMAX + ALPHA-BETA SEARCH
# ============================================================

def negamax(board, current_model, depth, alpha, beta, state):
    state["nodes"] += 1

    if board.is_checkmate():
        return -MATE_SCORE + state["ply"]

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
        or board.is_repetition(3)
    ):
        return 0

    if depth <= 0:
        return evaluate_board(board)

    if (
        state["nodes"] >= state["max_nodes"]
        or time.monotonic() >= state["deadline"]
    ):
        state["aborted"] = True
        return evaluate_board(board)

    moves = list(board.legal_moves)
    if not moves:
        return evaluate_board(board)

    logits = get_policy_logits(board, current_model)
    moves = order_moves(board, moves, logits)

    best_score = -float("inf")

    for move in moves:
        if (
            state["nodes"] >= state["max_nodes"]
            or time.monotonic() >= state["deadline"]
        ):
            state["aborted"] = True
            break

        board.push(move)
        state["ply"] += 1

        score = -negamax(
            board,
            current_model,
            depth - 1,
            -beta,
            -alpha,
            state,
        )

        state["ply"] -= 1
        board.pop()

        best_score = max(best_score, score)
        alpha = max(alpha, score)

        if alpha >= beta or state["aborted"]:
            break

    if best_score == -float("inf"):
        return evaluate_board(board)

    return best_score


def search_best_move(board: chess.Board, current_model, allowed_moves=None):
    legal_moves = list(board.legal_moves)

    if allowed_moves is not None:
        allowed_uci = {move.uci() for move in allowed_moves}
        legal_moves = [
            move for move in legal_moves if move.uci() in allowed_uci
        ]

    if not legal_moves:
        raise ValueError("No legal moves available for search.")

    if len(legal_moves) == 1:
        return legal_moves[0], {
            "depth": 0,
            "requested_depth": SEARCH_DEPTH,
            "nodes": 0,
            "elapsed_ms": 0,
            "aborted": False,
            "method": "forced_move",
        }

    started = time.monotonic()

    state = {
        "nodes": 0,
        "max_nodes": MAX_SEARCH_NODES,
        "deadline": started + SEARCH_TIME_LIMIT,
        "ply": 0,
        "aborted": False,
    }

    root_logits = get_policy_logits(board, current_model)
    ordered_root = order_moves(board, legal_moves, root_logits)
    best_move = ordered_root[0]
    best_score = -float("inf")
    completed_depth = 0

    for depth in range(1, SEARCH_DEPTH + 1):
        if (
            state["nodes"] >= state["max_nodes"]
            or time.monotonic() >= state["deadline"]
        ):
            state["aborted"] = True
            break

        iteration_move = None
        iteration_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")
        interrupted = False

        for move in ordered_root:
            if (
                state["nodes"] >= state["max_nodes"]
                or time.monotonic() >= state["deadline"]
            ):
                state["aborted"] = True
                interrupted = True
                break

            board.push(move)
            state["ply"] = 1

            score = -negamax(
                board,
                current_model,
                depth - 1,
                -beta,
                -alpha,
                state,
            )

            board.pop()
            state["ply"] = 0

            if score > iteration_score:
                iteration_score = score
                iteration_move = move

            alpha = max(alpha, score)

            if state["aborted"]:
                interrupted = True
                break

        if iteration_move is not None and not interrupted:
            best_move = iteration_move
            best_score = iteration_score
            completed_depth = depth
        else:
            break

    elapsed_ms = int((time.monotonic() - started) * 1000)

    return best_move, {
        "depth": completed_depth,
        "requested_depth": SEARCH_DEPTH,
        "nodes": state["nodes"],
        "elapsed_ms": elapsed_ms,
        "aborted": state["aborted"],
        "score": None if best_score == -float("inf") else int(best_score),
        "method": "NyraNet + Negamax/Alpha-Beta",
    }


# ============================================================
# REQUEST SCHEMA
# ============================================================

class MoveRequest(BaseModel):
    fen: str
    legal_moves: list[Any] = Field(default_factory=list)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "service": SERVICE_NAME,
        "version": "0.2.0",
        "health": "/health",
        "move": "/move",
        "debug_predict": "/debug/predict?fen=<fen>",
        "search": {
            "depth": SEARCH_DEPTH,
            "max_nodes": MAX_SEARCH_NODES,
            "time_limit_seconds": SEARCH_TIME_LIMIT,
        },
    }


@app.get("/health")
def health():
    with model_lock:
        return {
            "status": model_status,
            "model_loaded": model_loaded,
            "model_error": model_error,
        }


def parse_supplied_legal_moves(board: chess.Board, supplied):
    if not supplied:
        return None

    allowed = set()

    for item in supplied:
        if isinstance(item, str):
            raw_uci = item
        elif isinstance(item, dict):
            raw_uci = item.get("uci")
        else:
            continue

        if not isinstance(raw_uci, str) or not raw_uci:
            continue

        try:
            move = chess.Move.from_uci(raw_uci)
            if move in board.legal_moves:
                allowed.add(move.uci())
        except (ValueError, TypeError):
            continue

    if not allowed:
        raise HTTPException(
            status_code=400,
            detail="No valid legal_moves were supplied.",
        )

    return [
        move for move in board.legal_moves
        if move.uci() in allowed
    ]


@app.post("/move")
def choose_move(request: MoveRequest):
    with model_lock:
        current_model = model
        ready = model_loaded
        current_status = model_status
        current_error = model_error

    if not ready or current_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model is not ready",
                "model_status": current_status,
                "model_error": current_error,
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

    if board.is_game_over(claim_draw=True):
        raise HTTPException(
            status_code=400,
            detail="The game is already over.",
        )

    try:
        allowed_moves = parse_supplied_legal_moves(
            board, request.legal_moves
        )

        best_move, search_info = search_best_move(
            board,
            current_model,
            allowed_moves=allowed_moves,
        )

        return {
            "uci": best_move.uci(),
            "san": board.san(best_move),
            "search": search_info,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"MOVE ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Move search failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


@app.get("/debug/predict")
def debug_predict(fen: str):
    with model_lock:
        current_model = model
        ready = model_loaded
        current_error = model_error

    if not ready or current_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "model_loaded": ready,
                "model_error": current_error,
            },
        )

    try:
        board = chess.Board(fen)

        if not board.is_valid():
            raise HTTPException(
                status_code=400,
                detail="Invalid FEN",
            )

        x = encode_board(board).to(DEVICE)

        with torch.inference_mode():
            logits = current_model(x)[0]

        values, indices = torch.topk(logits, k=10)
        predictions = []

        for score, index in zip(values.tolist(), indices.tolist()):
            from_sq = index // 64
            to_sq = index % 64
            move = chess.Move(from_sq, to_sq)

            predictions.append({
                "uci": move.uci(),
                "score": round(float(score), 5),
                "is_legal": move in board.legal_moves,
            })

        legal_ranked = sorted(
            board.legal_moves,
            key=lambda move: float(
                logits[
                    move.from_square * 64 + move.to_square
                ].item()
            ),
            reverse=True,
        )[:10]

        return {
            "model_loaded": True,
            "prediction_source": "NyraNet raw logits",
            "top_predictions": predictions,
            "top_legal_predictions": [
                {
                    "uci": move.uci(),
                    "san": board.san(move),
                    "score": round(
                        float(
                            logits[
                                move.from_square * 64 + move.to_square
                            ].item()
                        ),
                        5,
                    ),
                }
                for move in legal_ranked
            ],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        )
