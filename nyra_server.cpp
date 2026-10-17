#include "chess.hpp"
#include "httplib.h"
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

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "NyraNet");
Ort::Session* session;
const char* input_names[] = {"input"};
const char* output_names[] = {"policy"};

// --- OPENING BOOK ---
std::unordered_map<std::string, std::string> opening_book = {
    {"r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq -", "a7a6"},
    {"r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -", "c2c3"},
    {"rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -", "d2d4"},
    {"rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq -", "b1c3"},
    {"rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq -", "d7d5"}
};

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

// --- HÀM ĐÁNH GIÁ ---
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
            int rank = sq / 8; int file = sq % 8;
            int dist = std::abs(3.5 - rank) + std::abs(3.5 - file);

            Square s = static_cast<Square>(sq);
            Color enemy = (piece.color() == Color::WHITE) ? Color::BLACK : Color::WHITE;
            if (board.isAttacked(s, enemy) && !board.isAttacked(s, piece.color()) && piece.type() != PieceType::KING) {
                val -= PIECE_VALUES[pt_idx] * 0.8; 
            }

            if (piece.type() == PieceType::PAWN) {
                val += (14 - 4 * dist);
                if (piece.color() == Color::WHITE) val += rank * 5; else val += (7 - rank) * 5;
            } else if (piece.type() == PieceType::KNIGHT || piece.type() == PieceType::BISHOP) {
                if (piece.color() == Color::WHITE && rank > 0) val += 30;
                if (piece.color() == Color::BLACK && rank < 7) val += 30;
                val += (14 - 4 * dist);
            }
            score += sign * val;
        }
    }
    return (board.sideToMove() == Color::WHITE) ? score : -score;
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

// Đã sửa hàm kiểm tra quân nặng để tương thích với thư viện mới
bool has_non_pawn_material(const Board& board, Color color) {
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.at(sq);
        if (p != Piece::NONE && p.color() == color) {
            PieceType pt = p.type();
            if (pt == PieceType::KNIGHT || pt == PieceType::BISHOP || pt == PieceType::ROOK || pt == PieceType::QUEEN)
                return true;
        }
    }
    return false;
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

// --- THUẬT TOÁN TÌM KIẾM NÂNG CAO ---
int negamax(Board& board, int depth, int alpha, int beta, bool root = false) {
    if (depth == 0) return quiescence(board, alpha, beta);

    bool in_check = board.inCheck();

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

    Movelist moves; movegen::legalmoves(moves, board);
    if (moves.size() == 0) return in_check ? -MATE_SCORE + depth : 0;
    
    order_moves_mvv_lva(board, moves);
    
    int best = -INF;
    Move best_move_in_node = moves[0];
    TTFlag flag = UPPERBOUND;

    for (int i = 0; i < moves.size(); i++) {
        board.makeMove(moves[i]);
        int extension = board.inCheck() ? 1 : 0;
        int score;
        int new_depth = depth - 1 + extension;

        // LMR (Late Move Reductions) - Sửa lại điều kiện so sánh với CheckType
        if (i >= 3 && depth >= 3 && !in_check && !board.isCapture(moves[i]) && board.givesCheck(moves[i]) == CheckType::NONE) {
            new_depth -= 1; 
        }

        // PVS (Principal Variation Search)
        if (i == 0) {
            score = -negamax(board, new_depth, -beta, -alpha);
        } else {
            score = -negamax(board, new_depth, -alpha - 1, -alpha);
            if (score > alpha && new_depth < depth - 1 + extension) {
                score = -negamax(board, depth - 1 + extension, -beta, -alpha);
            } else if (score > alpha && score < beta) {
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

std::string extract_fen(const std::string& body) {
    size_t start = body.find("\"fen\"");
    if (start == std::string::npos) return "";
    start = body.find("\"", start + 5); start++; 
    size_t end = body.find("\"", start);
    if (end == std::string::npos) return "";
    return body.substr(start, end - start);
}

int main() {
    init_zobrist();
    Ort::SessionOptions opts; opts.SetIntraOpNumThreads(1);
    std::cout << "Đang load mô hình ONNX..." << std::endl;
    session = new Ort::Session(env, "nyra_net.onnx", opts);

    httplib::Server svr;

    svr.Options(R"(/.*)", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 200;
    });

    svr.Post("/move", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        std::string fen = extract_fen(req.body);
        if (fen.empty()) {
            res.status = 400; res.set_content("{\"error\":\"No FEN\"}", "application/json"); return;
        }

        std::string fen_short = fen.substr(0, fen.find_last_of(" "));
        fen_short = fen_short.substr(0, fen_short.find_last_of(" ")); 
        
        Board board = Board(fen);
        Movelist moves; movegen::legalmoves(moves, board);
        
        if (moves.size() == 0) {
            res.set_content("{\"uci\":\"NONE\"}", "application/json"); return;
        }

        if (opening_book.find(fen_short) != opening_book.end()) {
            std::string book_move_uci = opening_book[fen_short];
            for (int i = 0; i < moves.size(); i++) {
                if (uci::moveToUci(moves[i]) == book_move_uci) {
                    std::string json_resp = "{\"uci\":\"" + book_move_uci + "\", \"search\":{\"depth\":0, \"method\":\"Opening Book\"}}";
                    res.set_content(json_resp, "application/json");
                    return;
                }
            }
        }

        transposition_table.clear();
        order_moves_mvv_lva(board, moves);
        
        int best_score = -INF;
        Move best_move = moves[0];
        int SEARCH_DEPTH = 6; 

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < moves.size(); i++) {
            board.makeMove(moves[i]);
            int score = -negamax(board, SEARCH_DEPTH - 1, -INF, -best_score);
            board.unmakeMove(moves[i]);
            if (score > best_score) { best_score = score; best_move = moves[i]; }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        int ms = (int)std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        std::string uci = uci::moveToUci(best_move);
        std::string json_resp = "{\"uci\":\"" + uci + "\", \"search\":{\"depth\":" + std::to_string(SEARCH_DEPTH) + 
                                ", \"time_ms\":" + std::to_string(ms) + "}}";
        
        std::cout << "FEN: " << fen << " -> " << uci << " (" << ms << "ms)" << std::endl;
        res.set_content(json_resp, "application/json");
    });

    std::cout << "Server C++ đang chạy ở cổng 8081..." << std::endl;
    svr.listen("0.0.0.0", 8081);
    delete session;
    return 0;
}
