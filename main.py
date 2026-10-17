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
# NYRA CHESS API 0.4.1
# - Fix: Khuyến khích Vua NHẬP THÀNH (thay vì giữ quyền nhập thành)
# - Fix: Ưu tiên bắt tốt qua đường (En passant) trong move ordering
# ============================================================

SERVICE_NAME = "Nyra Chess API"
VERSION = "0.4.1" # Đã tăng version

MODEL_URL = os.getenv("MODEL_URL", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "/tmp/nyra_chess_best.pt")
)

DEVICE = torch.device("cpu")
torch.set_num_threads(1)

SEARCH_DEPTH = max(
    1, min(4, int(os.getenv("SEARCH_DEPTH", "2")))
)
MAX_SEARCH_NODES = max(
    50, min(10000, int(os.getenv("MAX_SEARCH_NODES", "350")))
)
SEARCH_TIME_LIMIT = max(
    0.25, min(30.0, float(os.getenv("SEARCH_TIME_LIMIT", "3.0")))
)
QUIESCENCE_DEPTH = max(
    0, min(6, int(os.getenv("QUIESCENCE_DEPTH", "3")))
)

app = FastAPI(title=SERVICE_NAME, version=VERSION)

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

MATE_SCORE = 100_000
INF = 1_000_000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


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
        self.blocks = nn.ModuleList([ResidualBlock(64) for _ in range(4)])
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
# CHECKPOINT DOWNLOAD & LOADING
# ============================================================

def download_checkpoint():
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
        print(f"Using checkpoint: {MODEL_PATH}")
        return

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL is missing in the Render environment.")

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MODEL_PATH.with_suffix(".download")

    try:
        print("Downloading Nyra Chess checkpoint...")
        with requests.get(MODEL_URL, headers=headers, stream=True, timeout=(30, 300), allow_redirects=True) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError("Downloaded checkpoint is empty.")
        temp_path.replace(MODEL_PATH)
        print(f"Checkpoint downloaded: {MODEL_PATH.stat().st_size} bytes")
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must be a dictionary.")
    for key in ("model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise RuntimeError("Checkpoint has no model_state_dict or state_dict.")


def load_model():
    global model, model_loaded, model_error, model_status
    with model_lock:
        model_status = "loading"
        model_loaded = False
        model_error = None

    try:
        download_checkpoint()
        try:
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")

        state_dict = extract_state_dict(checkpoint)
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

        candidate = NyraNet().to(DEVICE)
        candidate.load_state_dict(state_dict, strict=True)
        candidate.eval()

        with torch.inference_mode():
            output = candidate(torch.zeros(1, 18, 8, 8, device=DEVICE))

        if tuple(output.shape) != (1, 4096):
            raise RuntimeError(f"Unexpected model output: {tuple(output.shape)}")

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
    threading.Thread(target=load_model, daemon=True, name="nyra-model-loader").start()


@app.on_event("startup")
def startup_event():
    start_model_loading()


# ============================================================
# BOARD ENCODING
# ============================================================

PIECE_TO_PLANE = {
    (chess.WHITE, chess.PAWN): 0, (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2, (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4, (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6, (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8, (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10, (chess.BLACK, chess.KING): 11,
}


def encode_board(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros((18, 8, 8), dtype=torch.float32)
    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.color, piece.piece_type)]
        planes[plane, chess.square_rank(square), chess.square_file(square)] = 1.0
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE): planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): planes[16, :, :] = 1.0
    if board.ep_square is not None:
        planes[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0
    return planes.unsqueeze(0)


# ============================================================
# EVALUATION & HEURISTICS
# ============================================================

def terminal_score(board: chess.Board, ply: int):
    if board.is_checkmate():
        return -MATE_SCORE + ply
    if (board.is_stalemate() or board.is_insufficient_material() or 
        board.is_fifty_moves() or board.is_repetition(3)):
        return 0
    return None


def center_bonus(square: chess.Square) -> int:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    distance = abs(3.5 - file_idx) + abs(3.5 - rank_idx)
    return int(14 - 4 * distance)


def evaluate_white_score(board: chess.Board) -> int:
    score = 0
    phase = 0
    phase_values = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
    for piece in board.piece_map().values():
        phase += phase_values.get(piece.piece_type, 0)

    # Material & positional
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        relative_rank = rank_idx if piece.color == chess.WHITE else 7 - rank_idx
        value = PIECE_VALUES[piece.piece_type]

        if piece.piece_type == chess.PAWN:
            value += relative_rank * 5
            if file_idx in (3, 4): value += 5
            enemy_pawns = board.pieces(chess.PAWN, not piece.color)
            same_or_adjacent_files = [f for f in (file_idx - 1, file_idx, file_idx + 1) if 0 <= f < 8]
            blocked_passage = False
            for enemy_square in enemy_pawns:
                ef = chess.square_file(enemy_square)
                er = chess.square_rank(enemy_square)
                enemy_relative_rank = er if piece.color == chess.BLACK else 7 - er
                if ef in same_or_adjacent_files and enemy_relative_rank > relative_rank:
                    blocked_passage = True
                    break
            if not blocked_passage:
                value += 8 + relative_rank * relative_rank * 2

        elif piece.piece_type == chess.KNIGHT:
            value += center_bonus(square)
        elif piece.piece_type == chess.BISHOP:
            value += center_bonus(square) // 2
        elif piece.piece_type == chess.ROOK:
            own_pawns = board.pieces(chess.PAWN, piece.color)
            enemy_pawns = board.pieces(chess.PAWN, not piece.color)
            own_on_file = any(chess.square_file(s) == file_idx for s in own_pawns)
            enemy_on_file = any(chess.square_file(s) == file_idx for s in enemy_pawns)
            if not own_on_file:
                value += 12 if not enemy_on_file else 6
            if relative_rank == 6: value += 12
        elif piece.piece_type == chess.QUEEN:
            if board.fullmove_number <= 10: value -= 8

        score += sign * value

    # Development in opening
    if board.fullmove_number <= 12 and phase >= 10:
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            home_rank = 0 if color == chess.WHITE else 7
            developed = 0
            for piece_type in (chess.KNIGHT, chess.BISHOP):
                for square in board.pieces(piece_type, color):
                    if chess.square_rank(square) != home_rank:
                        developed += 1
            score += sign * developed * 10

    # King safety (FIXED)
    endgame = phase <= 6
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        king_sq = board.king(color)
        if king_sq is None: continue

        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)
        home_rank = 0 if color == chess.WHITE else 7

        castled_position = (
            (color == chess.WHITE and kr == 0 and kf in (2, 6)) or
            (color == chess.BLACK and kr == 7 and kf in (2, 6))
        )

        # THƯỞNG LỚN NẾU ĐÃ NHẬP THÀNH AN TOÀN
        if castled_position:
            score += sign * 70

        # Phạt nếu vua đi lang thang ra khỏi hàng nhà (chưa nhập thành)
        if not endgame and board.fullmove_number <= 20:
            if kr != home_rank and not castled_position:
                score -= sign * 45

        if not endgame:
            direction = 1 if color == chess.WHITE else -1
            shield = 0
            for df in (-1, 0, 1):
                nf = kf + df
                nr = kr + direction
                if 0 <= nf < 8 and 0 <= nr < 8:
                    piece = board.piece_at(chess.square(nf, nr))
                    if piece and piece.color == color and piece.piece_type == chess.PAWN:
                        shield += 1
            score += sign * shield * 9

            if kf in (3, 4) and board.fullmove_number > 8:
                score -= sign * 20

            attackers = board.attackers(not color, king_sq)
            score -= sign * min(3, len(attackers)) * 14
        else:
            score += sign * center_bonus(king_sq)

    # Doubled pawns
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        pawns = board.pieces(chess.PAWN, color)
        for file_idx in range(8):
            count = sum(chess.square_file(sq) == file_idx for sq in pawns)
            if count > 1: score -= sign * (count - 1) * 12

    if board.is_check():
        score += -20 if board.turn == chess.WHITE else 20

    return score


def evaluate_board(board: chess.Board) -> int:
    terminal = terminal_score(board, 0)
    if terminal is not None: return terminal
    score = evaluate_white_score(board)
    return score if board.turn == chess.WHITE else -score


# ============================================================
# MOVE ORDERING
# ============================================================

def get_policy_logits(board: chess.Board, current_model):
    x = encode_board(board).to(DEVICE)
    with torch.inference_mode():
        return current_model(x)[0]


def move_capture_value(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move):
        return 0

    # En passant đặc biệt: quân bị bắt (tốt) không nằm ở ô đến
    if board.is_en_passant(move):
        victim_value = PIECE_VALUES[chess.PAWN]
        # Thưởng thêm điểm để AI không bỏ qua En passant
        return 15_000 + victim_value * 10
    else:
        victim = board.piece_at(move.to_square)
        victim_value = PIECE_VALUES[victim.piece_type] if victim else 0

    attacker = board.piece_at(move.from_square)
    attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0

    return 10_000 + victim_value * 10 - attacker_value


def order_moves(board: chess.Board, moves, logits):
    ranked = []
    for move in moves:
        index = move.from_square * 64 + move.to_square
        score = float(logits[index].item())
        if move.promotion:
            score += 5_000 + PIECE_VALUES.get(move.promotion, 0)
        score += move_capture_value(board, move)
        
        board.push(move)
        if board.is_check(): score += 100
        board.pop()
        ranked.append((score, move))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [move for _, move in ranked]


def order_moves_mvv_lva(board: chess.Board, moves):
    def key(move):
        s = 0.0
        if move.promotion:
            s += 5_000 + PIECE_VALUES.get(move.promotion, 0)
        s += move_capture_value(board, move)
        board.push(move)
        if board.is_check(): s += 100
        board.pop()
        return s
    return sorted(moves, key=key, reverse=True)


# ============================================================
# SEARCH ALGORITHMS
# ============================================================

def should_stop(state) -> bool:
    if state["nodes"] >= state["max_nodes"]:
        state["aborted"] = True
        return True
    if time.monotonic() >= state["deadline"]:
        state["aborted"] = True
        return True
    return False


def quiescence(board, alpha, beta, state, ply, remaining):
    state["nodes"] += 1
    terminal = terminal_score(board, ply)
    if terminal is not None: return terminal
    if should_stop(state): return evaluate_board(board)

    in_check = board.is_check()
    if not in_check:
        stand_pat = evaluate_board(board)
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
        if remaining <= 0: return stand_pat
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        if not moves: return stand_pat
    else:
        moves = list(board.legal_moves)
        if not moves: return -MATE_SCORE + ply
        if remaining <= 0: return evaluate_board(board)

    moves = order_moves_mvv_lva(board, moves)
    for move in moves:
        if should_stop(state): break
        board.push(move)
        score = -quiescence(board, -beta, -alpha, state, ply + 1, remaining - 1)
        board.pop()
        if score >= beta: return beta
        alpha = max(alpha, score)
    return alpha


def negamax(board, current_model, depth, alpha, beta, state, ply=0):
    state["nodes"] += 1
    terminal = terminal_score(board, ply)
    if terminal is not None: return terminal
    if should_stop(state): return evaluate_board(board)
    if depth <= 0:
        return quiescence(board, alpha, beta, state, ply, QUIESCENCE_DEPTH)

    moves = list(board.legal_moves)
    if not moves: return evaluate_board(board)

    if ply <= 1:
        logits = get_policy_logits(board, current_model)
        moves = order_moves(board, moves, logits)
    else:
        moves = order_moves_mvv_lva(board, moves)

    best_score = -INF
    for move in moves:
        if should_stop(state): break
        board.push(move)
        score = -negamax(board, current_model, depth - 1, -beta, -alpha, state, ply + 1)
        board.pop()
        best_score = max(best_score, score)
        alpha = max(alpha, score)
        if alpha >= beta: break

    if best_score == -INF: return evaluate_board(board)
    return best_score


def search_best_move(board, current_model, allowed_moves=None):
    legal_moves = list(board.legal_moves)
    if allowed_moves is not None:
        allowed_uci = {move.uci() for move in allowed_moves}
        legal_moves = [move for move in legal_moves if move.uci() in allowed_uci]

    if not legal_moves: raise ValueError("No legal moves available for search.")
    if len(legal_moves) == 1:
        return legal_moves[0], {"depth": 0, "requested_depth": SEARCH_DEPTH, "nodes": 0, "elapsed_ms": 0, "aborted": False, "score": None, "method": "forced_move"}

    started = time.monotonic()
    state = {"nodes": 0, "max_nodes": MAX_SEARCH_NODES, "deadline": started + SEARCH_TIME_LIMIT, "aborted": False}

    root_logits = get_policy_logits(board, current_model)
    ordered_root = order_moves(board, legal_moves, root_logits)

    best_move = ordered_root[0]
    best_score = None
    completed_depth = 0

    for depth in range(1, SEARCH_DEPTH + 1):
        if should_stop(state): break
        iteration_move = None
        iteration_score = -INF
        alpha = -INF
        beta = INF
        interrupted = False

        for move in ordered_root:
            if should_stop(state):
                interrupted = True
                break
            board.push(move)
            score = -negamax(board, current_model, depth - 1, -beta, -alpha, state, ply=1)
            board.pop()

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
        "score": None if best_score is None else int(best_score),
        "method": "NyraNet + Eval + Negamax/Alpha-Beta (v0.4.1)",
    }


# ============================================================
# API ENDPOINTS
# ============================================================

class MoveRequest(BaseModel):
    fen: str
    legal_moves: list[Any] = Field(default_factory=list)

@app.get("/")
def root():
    return {"service": SERVICE_NAME, "version": VERSION, "health": "/health", "move": "/move", "debug_predict": "/debug/predict?fen=<fen>", "search": {"depth": SEARCH_DEPTH, "max_nodes": MAX_SEARCH_NODES, "time_limit_seconds": SEARCH_TIME_LIMIT, "quiescence_depth": QUIESCENCE_DEPTH}}

@app.get("/health")
def health():
    with model_lock:
        return {"status": model_status, "model_loaded": model_loaded, "model_error": model_error}

def parse_supplied_legal_moves(board: chess.Board, supplied):
    if not supplied: return None
    allowed = set()
    for item in supplied:
        if isinstance(item, str): raw_uci = item
        elif isinstance(item, dict): raw_uci = item.get("uci")
        else: continue
        if not isinstance(raw_uci, str) or not raw_uci: continue
        try:
            move = chess.Move.from_uci(raw_uci)
            if move in board.legal_moves: allowed.add(move.uci())
        except (ValueError, TypeError): continue
    if not allowed: raise HTTPException(status_code=400, detail="No valid legal_moves were supplied.")
    return [move for move in board.legal_moves if move.uci() in allowed]

@app.post("/move")
def choose_move(request: MoveRequest):
    with model_lock:
        current_model, ready, current_status, current_error = model, model_loaded, model_status, model_error
    if not ready or current_model is None:
        raise HTTPException(status_code=503, detail={"error": "Model is not ready", "model_status": current_status, "model_error": current_error})

    try:
        board = chess.Board(request.fen)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}")
    if not board.is_valid(): raise HTTPException(status_code=400, detail="Invalid chess position.")
    if board.is_game_over(claim_draw=True): raise HTTPException(status_code=400, detail="The game is already over.")

    try:
        allowed_moves = parse_supplied_legal_moves(board, request.legal_moves)
        best_move, search_info = search_best_move(board, current_model, allowed_moves=allowed_moves)
        return {"uci": best_move.uci(), "san": board.san(best_move), "search": search_info}
    except HTTPException: raise
    except Exception as exc:
        print(f"MOVE ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=f"Move search failed: {type(exc).__name__}: {exc}")

@app.get("/debug/predict")
def debug_predict(fen: str):
    with model_lock:
        current_model, ready, current_error = model, model_loaded, model_error
    if not ready or current_model is None: raise HTTPException(status_code=503, detail={"model_loaded": ready, "model_error": current_error})
    try:
        board = chess.Board(fen)
        if not board.is_valid(): raise HTTPException(status_code=400, detail="Invalid FEN")
        logits = get_policy_logits(board, current_model)
        values, indices = torch.topk(logits, k=10)
        predictions = []
        for score, index in zip(values.tolist(), indices.tolist()):
            move = chess.Move(index // 64, index % 64)
            predictions.append({"uci": move.uci(), "score": round(float(score), 5), "is_legal": move in board.legal_moves})
        legal_ranked = sorted(board.legal_moves, key=lambda move: float(logits[move.from_square * 64 + move.to_square].item()), reverse=True)[:10]
        return {"model_loaded": True, "prediction_source": "NyraNet raw policy logits", "top_predictions": predictions, "top_legal_predictions": [{"uci": move.uci(), "san": board.san(move), "score": round(float(logits[move.from_square * 64 + move.to_square].item()), 5)} for move in legal_ranked]}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
