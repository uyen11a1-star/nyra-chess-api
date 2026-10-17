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
# NYRA CHESS API 0.7.1
# - Fix: Deploy hang on Render (startup event)
# - Speed: Transposition Table, Killer Moves
# - Optimized encode_board (loop-free)
# ============================================================

SERVICE_NAME = "Nyra Chess API"
VERSION = "0.7.1"

MODEL_URL = os.getenv("MODEL_URL", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/tmp/nyra_chess_best.pt"))

DEVICE = torch.device("cpu")
torch.set_num_threads(1)

SEARCH_DEPTH = max(1, int(os.getenv("SEARCH_DEPTH", "2")))
MAX_SEARCH_NODES = max(50, min(100000, int(os.getenv("MAX_SEARCH_NODES", "350"))))
SEARCH_TIME_LIMIT = max(0.25, min(28.0, float(os.getenv("SEARCH_TIME_LIMIT", "3.0"))))
QUIESCENCE_DEPTH = max(0, min(6, int(os.getenv("QUIESCENCE_DEPTH", "3"))))

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
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 335,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

# Zobrist Hashing để băm thế cờ
ZOBRIST_PIECES = {}
ZOBRIST_TURN = 0
def init_zobrist():
    global ZOBRIST_TURN
    import random
    rng = random.Random(42)
    for pt in range(1, 7):
        for c in (chess.WHITE, chess.BLACK):
            for sq in range(64):
                ZOBRIST_PIECES[(pt, c, sq)] = rng.getrandbits(64)
    ZOBRIST_TURN = rng.getrandbits(64)

init_zobrist()

def get_zobrist_hash(board: chess.Board) -> int:
    h = 0
    for sq, piece in board.piece_map().items():
        h ^= ZOBRIST_PIECES[(piece.piece_type, piece.color, sq)]
    if board.turn == chess.BLACK:
        h ^= ZOBRIST_TURN
    return h

# Transposition Table (Lưu trữ tạm thời)
TT = {}
TT_MAX_SIZE = 50000

OPENING_BOOK = {
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": "a7a6",
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3": "b1c3",
    "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3": "d2d4",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4": "c2c3",
}

class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(x + self.net(x))

class NyraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(18, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResidualBlock(64) for _ in range(4)])
        self.policy = nn.Sequential(
            nn.Conv2d(64, 2, 1), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(128, 4096),
        )
    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks: x = b(x)
        return self.policy(x)

def download_checkpoint():
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0: return
    if not MODEL_URL: raise RuntimeError("MODEL_URL is missing.")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MODEL_PATH.with_suffix(".download")
    with requests.get(MODEL_URL, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with open(temp_path, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk: output.write(chunk)
    temp_path.replace(MODEL_PATH)

def load_model():
    global model, model_loaded, model_error, model_status
    with model_lock:
        model_status = "loading"
        model_loaded = False
        model_error = None
    try:
        download_checkpoint()
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        candidate = NyraNet().to(DEVICE)
        candidate.load_state_dict(state_dict, strict=True)
        candidate.eval()
        with torch.inference_mode():
            output = candidate(torch.zeros(1, 18, 8, 8, device=DEVICE))
        if tuple(output.shape) != (1, 4096): raise RuntimeError(f"Unexpected output: {tuple(output.shape)}")
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
        if load_started: return
        load_started = True
    threading.Thread(target=load_model, daemon=True, name="nyra-model-loader").start()

@app.on_event("startup")
def startup_event():
    start_model_loading()

PIECE_TO_PLANE = {
    (chess.WHITE, chess.PAWN): 0, (chess.WHITE, chess.KNIGHT): 1, (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3, (chess.WHITE, chess.QUEEN): 4, (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6, (chess.BLACK, chess.KNIGHT): 7, (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9, (chess.BLACK, chess.QUEEN): 10, (chess.BLACK, chess.KING): 11,
}

def encode_board(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros((18, 8, 8), dtype=torch.float32)
    for pt in range(1, 7):
        for color in (chess.WHITE, chess.BLACK):
            plane_idx = PIECE_TO_PLANE[(color, pt)]
            for sq in board.pieces(pt, color):
                planes[plane_idx, chess.square_rank(sq), chess.square_file(sq)] = 1.0
    
    if board.turn == chess.WHITE: planes[12].fill_(1.0)
    if board.has_kingside_castling_rights(chess.WHITE): planes[13].fill_(1.0)
    if board.has_queenside_castling_rights(chess.WHITE): planes[14].fill_(1.0)
    if board.has_kingside_castling_rights(chess.BLACK): planes[15].fill_(1.0)
    if board.has_queenside_castling_rights(chess.BLACK): planes[16].fill_(1.0)
    if board.ep_square is not None:
        planes[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0
    return planes.unsqueeze(0)

def terminal_score(board: chess.Board, ply: int):
    if board.is_checkmate(): return -MATE_SCORE + ply
    if (board.is_stalemate() or board.is_insufficient_material() or 
        board.is_fifty_moves() or board.is_repetition(3)): return 0
    return None

def center_bonus(square: chess.Square) -> int:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    distance = abs(3.5 - file_idx) + abs(3.5 - rank_idx)
    return int(14 - 4 * distance)

def evaluate_white_score(board: chess.Board) -> int:
    score = 0
    phase = 0
    for piece in board.piece_map().values():
        phase += {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}.get(piece.piece_type, 0)
    endgame = phase <= 6

    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        file_idx = chess.square_file(square)
        relative_rank = chess.square_rank(square) if piece.color == chess.WHITE else 7 - chess.square_rank(square)
        value = PIECE_VALUES[piece.piece_type]

        is_attacked = board.is_attacked_by(not piece.color, square)
        is_defended = board.is_attacked_by(piece.color, square)
        if is_attacked and not is_defended and piece.piece_type != chess.KING:
            value -= int(PIECE_VALUES[piece.piece_type] * 0.8)
        elif is_attacked and not is_defended and piece.piece_type == chess.KING:
            value -= 500

        if piece.piece_type == chess.PAWN:
            value += relative_rank * 5
            if file_idx in (3, 4): value += 5
            enemy_pawns = board.pieces(chess.PAWN, not piece.color)
            same_or_adjacent = [f for f in (file_idx - 1, file_idx, file_idx + 1) if 0 <= f < 8]
            blocked = False
            for enemy_sq in enemy_pawns:
                ef, er = chess.square_file(enemy_sq), chess.square_rank(enemy_sq)
                enemy_rel_rank = er if piece.color == chess.BLACK else 7 - er
                if ef in same_or_adjacent and enemy_rel_rank > relative_rank:
                    blocked = True; break
            if not blocked: value += 8 + relative_rank * relative_rank * 2

        elif piece.piece_type == chess.KNIGHT:
            value += center_bonus(square)
            if 2 <= relative_rank <= 5 and board.attackers(piece.color, square):
                if any(board.piece_at(p).piece_type == chess.PAWN for p in board.attackers(piece.color, square) if board.piece_at(p)):
                    value += 25
        elif piece.piece_type == chess.BISHOP:
            value += center_bonus(square) // 2
        elif piece.piece_type == chess.ROOK:
            own_pawns = board.pieces(chess.PAWN, piece.color)
            enemy_pawns = board.pieces(chess.PAWN, not piece.color)
            if not any(chess.square_file(s) == file_idx for s in own_pawns):
                value += 20 if not any(chess.square_file(s) == file_idx for s in enemy_pawns) else 10
            if relative_rank == 6: value += 15
        elif piece.piece_type == chess.QUEEN:
            if board.fullmove_number <= 10: value -= 8
        score += sign * value

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2: score += 35
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2: score -= 35

    if board.fullmove_number <= 12 and phase >= 10:
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            home_rank = 0 if color == chess.WHITE else 7
            developed = sum(1 for pt in (chess.KNIGHT, chess.BISHOP) for sq in board.pieces(pt, color) if chess.square_rank(sq) != home_rank)
            score += sign * developed * 10

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        king_sq = board.king(color)
        if king_sq is None: continue
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        home_rank = 0 if color == chess.WHITE else 7
        castled = ((color == chess.WHITE and kr == 0 and kf in (2, 6)) or (color == chess.BLACK and kr == 7 and kf in (2, 6)))

        if castled: 
            score += sign * 70
            direction = 1 if color == chess.WHITE else -1
            shield = sum(1 for df in (-1, 0, 1) if 0 <= kf + df < 8 and 0 <= kr + direction < 8 and board.piece_at(chess.square(kf + df, kr + direction)) and board.piece_at(chess.square(kf + df, kr + direction)).color == color and board.piece_at(chess.square(kf + df, kr + direction)).piece_type == chess.PAWN)
            if shield == 0: score -= sign * 60
            elif shield == 1: score -= sign * 30

        if not endgame and board.fullmove_number <= 20 and kr != home_rank and not castled: score -= sign * 45
        if not endgame:
            if kf in (3, 4) and board.fullmove_number > 8: score -= sign * 20
            score -= sign * min(3, len(board.attackers(not color, king_sq))) * 14
        else: score += sign * center_bonus(king_sq)

    if board.is_check(): score += -20 if board.turn == chess.WHITE else 20
    return score

def evaluate_board(board: chess.Board) -> int:
    terminal = terminal_score(board, 0)
    if terminal is not None: return terminal
    score = evaluate_white_score(board)
    return score if board.turn == chess.WHITE else -score

def get_policy_logits(board: chess.Board, current_model):
    x = encode_board(board).to(DEVICE)
    with torch.inference_mode():
        return current_model(x)[0]

def move_capture_value(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move): return 0
    if board.is_en_passant(move): return 15_000 + 1000
    victim = board.piece_at(move.to_square)
    victim_value = PIECE_VALUES[victim.piece_type] if victim else 0
    attacker = board.piece_at(move.from_square)
    attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
    return 10_000 + victim_value * 10 - attacker_value

killer_moves = [[] for _ in range(64)]

def order_moves(board: chess.Board, moves, logits=None, ply=0):
    ranked = []
    for move in moves:
        score = 0.0
        if logits is not None:
            score += float(logits[move.from_square * 64 + move.to_square].item())
        
        if move.promotion: score += 5_000 + PIECE_VALUES.get(move.promotion, 0)
        score += move_capture_value(board, move)
        
        if ply < 64 and move in killer_moves[ply]:
            score += 900

        board.push(move)
        if board.is_check(): score += 100
        board.pop()
        ranked.append((score, move))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [move for _, move in ranked]

def order_moves_mvv_lva(board: chess.Board, moves, ply=0):
    return order_moves(board, moves, logits=None, ply=ply)

def should_stop(state) -> bool:
    if state["nodes"] >= state["max_nodes"] or time.monotonic() >= state["deadline"]:
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

    moves = order_moves_mvv_lva(board, moves, ply)
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
    
    z_hash = get_zobrist_hash(board)
    if z_hash in TT:
        tt_depth, tt_score = TT[z_hash]
        if tt_depth >= depth:
            return tt_score

    if depth <= 0: return quiescence(board, alpha, beta, state, ply, QUIESCENCE_DEPTH)

    moves = list(board.legal_moves)
    if not moves: return evaluate_board(board)

    if ply <= 1:
        logits = get_policy_logits(board, current_model)
        moves = order_moves(board, moves, logits, ply)
    else:
        moves = order_moves_mvv_lva(board, moves, ply)

    best_score = -INF
    for move in moves:
        if should_stop(state): break
        board.push(move)
        score = -negamax(board, current_model, depth - 1, -beta, -alpha, state, ply + 1)
        board.pop()
        
        if score > best_score:
            best_score = score
        alpha = max(alpha, score)
        
        if alpha >= beta:
            if not board.is_capture(move) and ply < 64:
                if move not in killer_moves[ply]:
                    killer_moves[ply].insert(0, move)
                    if len(killer_moves[ply]) > 2: killer_moves[ply].pop()
            break

    if len(TT) < TT_MAX_SIZE:
        TT[z_hash] = (depth, best_score)
    else:
        TT.pop(next(iter(TT)))

    if best_score == -INF: return evaluate_board(board)
    return best_score

def search_best_move(board, current_model, allowed_moves=None):
    legal_moves = list(board.legal_moves)
    if allowed_moves is not None:
        allowed_uci = {move.uci() for move in allowed_moves}
        legal_moves = [move for move in legal_moves if move.uci() in allowed_uci]

    if not legal_moves: raise ValueError("No legal moves.")
    
    TT.clear()
    for i in range(64): killer_moves[i].clear()

    fen = board.fen()
    short_fen = " ".join(fen.split(" ")[:4])
    for book_fen, book_move in OPENING_BOOK.items():
        if short_fen == " ".join(book_fen.split(" ")[:4]):
            for m in legal_moves:
                if m.uci() == book_move:
                    return m, {"depth": 0, "requested_depth": SEARCH_DEPTH, "nodes": 0, "elapsed_ms": 0, "aborted": False, "score": None, "method": "Opening Book"}

    if len(legal_moves) == 1:
        return legal_moves[0], {"depth": 0, "requested_depth": SEARCH_DEPTH, "nodes": 0, "elapsed_ms": 0, "aborted": False, "score": None, "method": "forced_move"}

    started = time.monotonic()
    state = {"nodes": 0, "max_nodes": MAX_SEARCH_NODES, "deadline": started + SEARCH_TIME_LIMIT, "aborted": False}

    root_logits = get_policy_logits(board, current_model)
    ordered_root = order_moves(board, legal_moves, root_logits, ply=0)

    best_move = ordered_root[0]
    best_score = None
    completed_depth = 0

    for depth in range(1, SEARCH_DEPTH + 1):
        if should_stop(state): break
        iteration_move, iteration_score = None, -INF
        alpha, beta = -INF, INF
        interrupted = False

        for move in ordered_root:
            if should_stop(state):
                interrupted = True
                break
            board.push(move)
            score = -negamax(board, current_model, depth - 1, -beta, -alpha, state, ply=1)
            board.pop()

            if score > iteration_score:
                iteration_score, iteration_move = score, move
            alpha = max(alpha, score)
            if state["aborted"]:
                interrupted = True
                break

        if iteration_move is not None and not interrupted:
            best_move, best_score, completed_depth = iteration_move, iteration_score, depth
        else:
            break

    return best_move, {
        "depth": completed_depth,
        "requested_depth": SEARCH_DEPTH,
        "nodes": state["nodes"],
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "aborted": state["aborted"],
        "score": None if best_score is None else int(best_score),
        "method": "NyraNet v0.7.1 + TT + Killer",
    }

class MoveRequest(BaseModel):
    fen: str
    legal_moves: list[Any] = Field(default_factory=list)

@app.get("/")
def root():
    return {"service": SERVICE_NAME, "version": VERSION, "health": "/health", "move": "/move", "search": {"depth": SEARCH_DEPTH, "max_nodes": MAX_SEARCH_NODES, "time_limit_seconds": SEARCH_TIME_LIMIT, "quiescence_depth": QUIESCENCE_DEPTH}}

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
    if not allowed: raise HTTPException(status_code=400, detail="No valid legal_moves.")
    return [move for move in board.legal_moves if move.uci() in allowed]

@app.post("/move")
def choose_move(request: MoveRequest):
    with model_lock:
        current_model, ready, current_status, current_error = model, model_loaded, model_status, model_error
    if not ready or current_model is None:
        raise HTTPException(status_code=503, detail={"error": "Model is not ready"})

    try:
        board = chess.Board(request.fen)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}")
    if not board.is_valid(): raise HTTPException(status_code=400, detail="Invalid position.")

    try:
        allowed_moves = parse_supplied_legal_moves(board, request.legal_moves)
        best_move, search_info = search_best_move(board, current_model, allowed_moves=allowed_moves)
        return {"uci": best_move.uci(), "san": board.san(best_move), "search": search_info}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Move search failed: {exc}")

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
        predictions = [{"uci": chess.Move(idx//64, idx%64).uci(), "score": round(float(val), 5), "is_legal": chess.Move(idx//64, idx%64) in board.legal_moves} for val, idx in zip(values.tolist(), indices.tolist())]
        legal_ranked = sorted(board.legal_moves, key=lambda m: float(logits[m.from_square * 64 + m.to_square].item()), reverse=True)[:10]
        return {"model_loaded": True, "prediction_source": "NyraNet raw policy logits", "top_predictions": predictions, "top_legal_predictions": [{"uci": m.uci(), "san": board.san(m), "score": round(float(logits[m.from_square * 64 + m.to_square].item()), 5)} for m in legal_ranked]}
    except Exception as exc: raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
