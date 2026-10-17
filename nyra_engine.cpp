#include "chess.hpp"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <random>

using namespace chess;

const int PIECE_VALUES[7] = {0, 100, 320, 330, 500, 900, 20000}; 
const int MATE_SCORE = 100000;
const int INF = 200000;

// --- PIECE SQUARE TABLES (PST) ---
const int PAWN_PST[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};
const int KNIGHT_PST[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};
const int BISHOP_PST[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};
const int ROOK_PST[64] = {
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
};
const int QUEEN_PST[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};
const int KING_MID_PST[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
};

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "NyraNet");
Ort::Session* session;
const char* input_names[] = {"input"};
const char* output_names[] = {"policy"};

// --- ZOBRIST HASHING ---
uint64_t zobrist_pieces[7][2][64];
uint64_t zobrist_turn;
enum TTFlag { EXACT, LOWERBOUND, UPPERBOUND };
struct TTEntry { int depth; int score; TTFlag flag; Move best_move; };
std::unordered_map<uint64_t, TTEntry> transposition_table;

void init_zobrist() {
    std::mt19937_64 rng(12345);
    for (int pt = 1; pt <= 6; pt++)
        for (int c = 0; c < 2; c++)
            for (int sq = 0; sq < 64; sq++)
                zobrist_pieces[pt][c][sq] = rng();
    zobrist_turn = rng();
}

uint64_t get_hash(const Board& board) {
    uint64_t h = 0;
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.at(sq);
        if (p != Piece::NONE) {
            int c = (p.color() == Color::WHITE) ? 0 : 1;
            h ^= zobrist_pieces[static_cast<int>(p.type())][c][sq];
        }
    }
    if (board.sideToMove() == Color::WHITE) h ^= zobrist_turn;
    return h;
}

// --- MÃ HÓA ĐƠN GIẢN (Bỏ qua quyền nhập thành) ---
void encode_board(const Board& board, std::vector<float>& tensor) {
    std::fill(tensor.begin(), tensor.end(), 0.0f);
    for (int sq = 0; sq < 64; sq++) {
        Piece piece = board.at(sq);
        if (piece != Piece::NONE) {
            int plane = -1; int pt = static_cast<int>(piece.type());
            if (piece.color() == Color::WHITE) plane = pt - 1;       
            else plane = (pt - 1) + 6;            
            tensor[plane * 64 + (sq / 8) * 8 + (sq % 8)] = 1.0f;
        }
    }
    if (board.sideToMove() == Color::WHITE) {
        for (int i = 0; i < 64; i++) tensor[12 * 64 + i] = 1.0f;
    }
    // Tạm thời bỏ qua plane 13-17 (castling, en passant)
}

// --- HÀM ĐÁNH GIÁ CÓ BẢNG ĐIỂM VỊ TRÍ (PST) ---
int evaluate_board(const Board& board) {
    if (board.isGameOver().second == GameResult::LOSE) return -MATE_SCORE;
    if (board.isGameOver().second == GameResult::DRAW) return 0;

    int score = 0;
    for (int sq = 0; sq < 64; sq++) {
        Piece piece = board.at(sq);
        if (piece != Piece::NONE) {
            int sign = (piece.color() == Color::WHITE) ? 1 : -1;
            int pt_idx = static_cast<int>(piece.type());
            int val = PIECE_VALUES[pt_idx];
            int pst_sq = (piece.color() == Color::WHITE) ? sq : (sq ^ 56);

            if (piece.type() == PieceType::PAWN) val += PAWN_PST[pst_sq];
            else if (piece.type() == PieceType::KNIGHT) val += KNIGHT_PST[pst_sq];
            else if (piece.type() == PieceType::BISHOP) val += BISHOP_PST[pst_sq];
            else if (piece.type() == PieceType::ROOK) val += ROOK_PST[pst_sq];
            else if (piece.type() == PieceType::QUEEN) val += QUEEN_PST[pst_sq];
            else if (piece.type() == PieceType::KING) val += KING_MID_PST[pst_sq];

            Square s = static_cast<Square>(sq);
            Color enemy = (piece.color() == Color::WHITE) ? Color::BLACK : Color::WHITE;
            if (board.isAttacked(s, enemy) && !board.isAttacked(s, piece.color()) && piece.type() != PieceType::KING) {
                val -= PIECE_VALUES[pt_idx] * 0.8; 
            }
            score += sign * val;
        }
    }
    return (board.sideToMove() == Color::WHITE) ? score : -score;
}

void order_moves_with_nn(Board& board, Movelist& moves) {
    std::vector<float> inputs(18 * 8 * 8);
    encode_board(board, inputs);
    std::array<int64_t, 4> shape{1, 18, 8, 8};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, inputs.data(), inputs.size(), shape.data(), shape.size());
    auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
    float* logits = outputs.front().GetTensorMutableData<float>();

    struct SM { Move m; float s; };
    std::vector<SM> sm;
    for (int i = 0; i < moves.size(); i++) {
        int idx = moves[i].from().index() * 64 + moves[i].to().index();
        float s = logits[idx] + (board.isCapture(moves[i]) ? 1000.0f : 0.0f); 
        sm.push_back({moves[i], s});
    }
    std::sort(sm.begin(), sm.end(), [](const SM& a, const SM& b) { return a.s > b.s; });
    for (int i = 0; i < moves.size(); i++) moves[i] = sm[i].m;
}

void order_moves_mvv_lva(Board& board, Movelist& moves) {
    struct SM { Move m; int s; };
    std::vector<SM> sm;
    for (int i = 0; i < moves.size(); i++) {
        int score = 0;
        if (board.isCapture(moves[i])) {
            Piece victim = board.at(moves[i].to());
            if (victim != Piece::NONE) score = 10 * PIECE_VALUES[static_cast<int>(victim.type())] - PIECE_VALUES[static_cast<int>(board.at(moves[i].from()).type())];
            else score = 1000;
        }
        sm.push_back({moves[i], score});
    }
    std::sort(sm.begin(), sm.end(), [](const SM& a, const SM& b) { return a.s > b.s; });
    for (int i = 0; i < moves.size(); i++) moves[i] = sm[i].m;
}

int quiescence(Board& board, int alpha, int beta) {
    int stand_pat = evaluate_board(board);
    if (stand_pat >= beta) return beta;
    if (alpha < stand_pat) alpha = stand_pat;

    Movelist moves; movegen::legalmoves(moves, board);
    Movelist capture_moves;
    for(int i=0; i<moves.size(); i++) {
        if(board.isCapture(moves[i])) capture_moves.add(moves[i]);
    }
    order_moves_mvv_lva(board, capture_moves);

    for (int i = 0; i < capture_moves.size(); i++) {
        board.makeMove(capture_moves[i]);
        int score = -quiescence(board, -beta, -alpha);
        board.unmakeMove(capture_moves[i]);
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

int negamax(Board& board, int depth, int alpha, int beta, bool root = false) {
    uint64_t hash = get_hash(board);
    if (transposition_table.find(hash) != transposition_table.end()) {
        TTEntry entry = transposition_table[hash];
        if (entry.depth >= depth) {
            if (entry.flag == EXACT) return entry.score;
            else if (entry.flag == LOWERBOUND && entry.score > alpha) alpha = entry.score;
            else if (entry.flag == UPPERBOUND && entry.score < beta) beta = entry.score;
            if (alpha >= beta) return entry.score;
        }
    }

    if (depth == 0) return quiescence(board, alpha, beta);

    Movelist moves; movegen::legalmoves(moves, board);
    if (moves.size() == 0) return board.inCheck() ? -MATE_SCORE + depth : 0;
    
    if (root) order_moves_with_nn(board, moves);
    else order_moves_mvv_lva(board, moves);
    
    int best = -INF;
    Move best_move_in_node = moves[0];
    TTFlag flag = UPPERBOUND;

    for (int i = 0; i < moves.size(); i++) {
        board.makeMove(moves[i]);
        int extension = board.inCheck() ? 1 : 0;
        int score;

        if (i == 0) {
            score = -negamax(board, depth - 1 + extension, -beta, -alpha);
        } else {
            score = -negamax(board, depth - 1 + extension, -alpha - 1, -alpha);
            if (score > alpha && score < beta) {
                score = -negamax(board, depth - 1 + extension, -beta, -alpha);
            }
        }
        board.unmakeMove(moves[i]);
        
        if (score > best) {
            best = score;
            best_move_in_node = moves[i];
        }
        if (score > alpha) {
            alpha = score;
            flag = EXACT;
        }
        if (alpha >= beta) {
            flag = LOWERBOUND;
            break;
        }
    }

    if (transposition_table.find(hash) == transposition_table.end() || transposition_table[hash].depth <= depth) {
        transposition_table[hash] = {depth, best, flag, best_move_in_node};
    }
    
    return best;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) return 1;
        std::string fen = argv[1];
        init_zobrist();
        
        Ort::SessionOptions opts; opts.SetIntraOpNumThreads(1);
        session = new Ort::Session(env, "nyra_net.onnx", opts);

        Board board = Board(fen);
        Movelist moves; movegen::legalmoves(moves, board);
        
        if (moves.size() == 0) {
            std::cout << "NONE" << std::endl;
            delete session;
            return 0;
        }

        transposition_table.clear();
        order_moves_with_nn(board, moves);
        
        int best_score = -INF;
        Move best_move = moves[0];
        int SEARCH_DEPTH = 5; 

        for (int i = 0; i < moves.size(); i++) {
            board.makeMove(moves[i]);
            int score = -negamax(board, SEARCH_DEPTH - 1, -INF, -best_score);
            board.unmakeMove(moves[i]);
            if (score > best_score) { best_score = score; best_move = moves[i]; }
        }

        std::cout << uci::moveToUci(best_move) << std::endl;
        delete session;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 2;
    }
}
