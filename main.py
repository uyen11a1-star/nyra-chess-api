import os
import time
import threading
import math
import random
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
# NYRA CHESS API 1.0.0 (MCTS Reasoning Engine)
# - Thay thế Alpha-Beta bằng Monte Carlo Tree Search (MCTS)
# - MCTS giúp AI "tưởng tượng" và suy luận chiến lược
# ============================================================

SERVICE_NAME = "Nyra Chess API"
VERSION = "1.0.0"

MODEL_URL = os.getenv("MODEL_URL", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/tmp/nyra_chess_best.pt"))

DEVICE = torch.device("cpu")
torch.set_num_threads(1)

# MCTS dùng thời gian để chạy mô phỏng, không dùng Depth
SEARCH_TIME_LIMIT = max(1.0, min(28.0, float(os.getenv("SEARCH_TIME_LIMIT", "10.0"))))
MCTS_SIMULATIONS = max(100, min(5000, int(os.getenv("MCTS_SIMULATIONS", "800"))))

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
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 335,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

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

def get_policy_logits(board: chess.Board, current_model):
    x = encode_board(board).to(DEVICE)
    with torch.inference_mode():
        return current_model(x)[0]

# ============================================================
# HEURISTIC EVALUATION (Dùng để chấm điểm cuối ván mô phỏng)
# ============================================================
def evaluate_board(board: chess.Board) -> int:
    if board.is_checkmate(): return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3): return 0

    score = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        file_idx = chess.square_file(square)
        relative_rank = chess.square_rank(square) if piece.color == chess.WHITE else 7 - chess.square_rank(square)
        value = PIECE_VALUES[piece.piece_type]

        is_attacked = board.is_attacked_by(not piece.color, square)
        is_defended = board.is_attacked_by(piece.color, square)
        if is_attacked and not is_defended and piece.piece_type != chess.KING:
            value -= int(PIECE_VALUES[piece.piece_type] * 0.8)

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
            value += int(14 - 4 * (abs(3.5 - file_idx) + abs(3.5 - relative_rank)))
        elif piece.piece_type == chess.BISHOP:
            value += int(7 - 2 * (abs(3.5 - file_idx) + abs(3.5 - relative_rank)))
        score += sign * value

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2: score += 35
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2: score -= 35

    if board.is_check(): score += -20 if board.turn == chess.WHITE else 20
    return score if board.turn == chess.WHITE else -score

# ============================================================
# MONTE CARLO TREE SEARCH (MCTS)
# ============================================================
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def is_fully_expanded(self):
        return self.is_expanded

    def best_child(self, c_param=1.414):
        choices_weights = [
            (child.value_sum / child.visits) + c_param * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self, current_model):
        if self.is_expanded: return
        legal_moves = list(self.board.legal_moves)
        if not legal_moves: return

        # Dùng mạng Neural để lấy xác suất (prior) cho các nước đi
        logits = get_policy_logits(self.board, current_model).cpu()
        probs = torch.softmax(logits, dim=0)

        for move in legal_moves:
            idx = move.from_square * 64 + move.to_square
            prior_prob = float(probs[idx].item())
            new_board = self.board.copy()
            new_board.push(move)
            self.children.append(MCTSNode(new_board, parent=self, move=move, prior=prior_prob))
        
        self.is_expanded = True

def run_mcts(root_board: chess.Board, current_model, time_limit):
    root = MCTSNode(root_board)
    start_time = time.time()
    simulations = 0

    while time.time() - start_time < time_limit and simulations < MCTS_SIMULATIONS:
        node = root
        
        # 1. Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            
        # 2. Expansion
        if not node.board.is_game_over() and not node.is_fully_expanded():
            node.expand(current_model)
            if node.children:
                node = node.children[0]

        # 3. Simulation (Rollout)
        temp_board = node.board.copy()
        depth = 0
        while not temp_board.is_game_over() and depth < 20:
            moves = list(temp_board.legal_moves)
            if not moves: break
            # Đi ngẫu nhiên có ưu tiên nước ăn quân
            captures = [m for m in moves if temp_board.is_capture(m)]
            move = random.choice(captures) if captures and random.random() < 0.5 else random.choice(moves)
            temp_board.push(move)
            depth += 1

        # 4. Backpropagation
        result = evaluate_board(temp_board)
        while node is not None:
            node.visits += 1
            if node.board.turn == chess.WHITE:
                node.value_sum += result
            else:
                node.value_sum -= result
            node = node.parent
        
        simulations += 1

    # Chọn nước đi có số lần duyệt (visits) nhiều nhất
    best_move = None
    max_visits = -1
    for child in root.children:
        if child.visits > max_visits:
            max_visits = child.visits
            best_move = child.move

    return best_move, simulations

# ============================================================
# ENDPOINT
# ============================================================
class MoveRequest(BaseModel):
    fen: str
    legal_moves: list[Any] = Field(default_factory=list)

@app.get("/")
def root():
    return {"service": SERVICE_NAME, "version": VERSION, "health": "/health", "move": "/move", "search": {"type": "MCTS", "time_limit_seconds": SEARCH_TIME_LIMIT, "max_simulations": MCTS_SIMULATIONS}}

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
        if allowed_moves is not None:
            board = chess.Board(request.fen)
            # Lọc nước đi nếu frontend gửi lên
            valid_moves = [m for m in board.legal_moves if m.uci() in [x.uci() for x in allowed_moves]]
            if not valid_moves: raise ValueError("No legal moves.")
        else:
            valid_moves = list(board.legal_moves)

        if len(valid_moves) == 1:
            return {"uci": valid_moves[0].uci(), "san": board.san(valid_moves[0]), "search": {"type": "forced_move"}}

        # Kiểm tra Opening Book
        fen = board.fen()
        short_fen = " ".join(fen.split(" ")[:4])
        for book_fen, book_move in OPENING_BOOK.items():
            if short_fen == " ".join(book_fen.split(" ")[:4]):
                for m in valid_moves:
                    if m.uci() == book_move:
                        return {"uci": m.uci(), "san": board.san(m), "search": {"type": "Opening Book"}}

        # Chạy MCTS
        best_move, simulations = run_mcts(board, current_model, SEARCH_TIME_LIMIT)
        
        if best_move is None:
            # Fallback nếu MCTS lỗi
            best_move = random.choice(valid_moves)
            simulations = 0

        return {
            "uci": best_move.uci(), 
            "san": board.san(best_move), 
            "search": {
                "type": "MCTS",
                "simulations": simulations,
                "time_limit": SEARCH_TIME_LIMIT
            }
        }
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
