#include "chess.hpp"
#include "httplib.h"
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

// --- BẢNG ĐIỂM VỊ TRÍ (PST) ---
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

// --- MỞ RỘNG OPENING BOOK ---
std::unordered_map<std::string, std::string> opening_book = {
    // 1. e4 e5
    {"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -", "e7e5"},
    // King's Gambit
    {"rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq f3", "e5f4"},
    // Italian Game
    {"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq -", "f8c5"},
    // Ruy Lopez
    {"r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq -", "a7a6"},
    // 1. e4 c5 (Sicilian)
    {"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -", "c7c5"},
    // Sicilian Open
    {"rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -", "d2d4"},
    // 1. e4 e6 (French)
    {"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -", "e7e6"},
    // 1. e4 c6 (Caro-Kann)
    {"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -", "c7c6"},
    // 1. d4 d5
    {"rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -", "d7d5"},
    // Queen's Gambit
    {"rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3", "e7e6"},
    // Queen's Gambit Accepted
    {"rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq -", "e2e3"}
};

// --- ZOBRIST HASHING ---
uint64_t zobrist_pieces[7][2][64];
uint64_t zobrist_turn;
uint64_t zobrist_castling[16];
uint64_t zobrist_ep[8];

enum TTFlag { EXACT, LOWERBOUND, UPPERBOUND };
struct TTEntry { int depth; int score; TTFlag flag; Move best_move; };
std::unordered_map<uint64_t, TTEntry> transposition_table;

std::chrono::steady_clock::time_point start_time;
bool time_is_up = false;
long long total_nodes = 0;

const int TIME_LIMIT_MS = 15000; 
const int MAX_DEPTH = 10; 
const int MAX_Q_DEPTH = 6; 

void init_zobrist() {
    std::mt19937_64 rng(12345);
    for (int pt = 1; pt <= 6; pt++)
        for (int c = 0; c < 2; c++)
            for (int sq = 0; sq < 64; sq++)
                zobrist_pieces[pt][c][sq] = rng();
    zobrist_turn = rng();
    for (int i = 0; i < 16; i++) zobrist_castling[i] = rng();
    for (int i = 0; i < 8; i++) zobrist_ep[i] = rng();
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
    
    std::string fen = board.getFen();
    size_t space1 = fen.find(' ');
    if (space1 != std::string::npos) {
        size_t space2 = fen.find(' ', space1 + 1);
        if (space2 != std::string::npos) {
            std::string castling = fen.substr(space1 + 1, space2 - space1 - 1);
            if (castling.find('K') != std::string::npos) h ^= zobrist_castling[1];
            if (castling.find('Q') != std::string::npos) h ^= zobrist_castling[2];
            if (castling.find('k') != std::string::npos) h ^= zobrist_castling[4];
            if (castling.find('q') != std::string::npos) h ^= zobrist_castling[8];
            
            size_t space3 = fen.find(' ', space2 + 1);
            if (space3 != std::string::npos && space3 > space2 + 1) {
                std::string ep = fen.substr(space2 + 1, space3 - space2 - 1);
                if (ep != "-") {
                    int ep_file = ep[0] - 'a';
                    h ^= zobrist_ep[ep_file];
                }
            }
        }
    }
    return h;
}

// ============================================================
// HÀM ĐÁNH GIÁ (SỬA LỖI XE, HẬU VÀ NHẬP THÀNH)
// ============================================================
int evaluate_white_score(Board& board) {
    int score = 0;
    int phase = 0;
    
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.at(sq);
        if (p != Piece::NONE) {
            PieceType pt = p.type();
            if (pt == PieceType::KNIGHT || pt == PieceType::BISHOP) phase += 1;
            else if (pt == PieceType::ROOK) phase += 2;
            else if (pt == PieceType::QUEEN) phase += 4;
        }
    }
    bool endgame = (phase <= 6);

    // 1. Material, PST & Tấn công Vua
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.at(sq);
        if (p == Piece::NONE) continue;

        int sign = (p.color() == Color::WHITE) ? 1 : -1;
        int pt_idx = static_cast<int>(p.type());
        int val = PIECE_VALUES[pt_idx];
        
        int pst_sq = (p.color() == Color::WHITE) ? sq : (sq ^ 56);
        if (p.type() == PieceType::PAWN) val += PAWN_PST[pst_sq];
        else if (p.type() == PieceType::KNIGHT) val += KNIGHT_PST[pst_sq];
        else if (p.type() == PieceType::BISHOP) val += BISHOP_PST[pst_sq];
        else if (p.type() == PieceType::ROOK) val += ROOK_PST[pst_sq];
        else if (p.type() == PieceType::QUEEN) val += QUEEN_PST[pst_sq];
        else if (p.type() == PieceType::KING) val += KING_MID_PST[pst_sq];

        Square s = static_cast<Square>(sq);
        Color enemy = (p.color() == Color::WHITE) ? Color::BLACK : Color::WHITE;

        bool is_attacked = board.isAttacked(s, enemy);
        bool is_defended = board.isAttacked(s, p.color());
        
        // Phạt quân bị treo (Hanging piece)
        if (is_attacked && !is_defended && p.type() != PieceType::KING) {
            val -= static_cast<int>(PIECE_VALUES[pt_idx] * 0.9);
        }

        // SỬA LỖI BỎ QUÊN HẬU: Hậu bị tấn công (dù có bảo vệ) -> Phạt nặng
        if (p.type() == PieceType::QUEEN && is_attacked) {
            val -= 100; 
        }

        // SỬA LỖI ĐI XE VÔ NGHĨA: Phạt Xe ra rìa hoặc đi lùi trong khai cuộc
        if (p.type() == PieceType::ROOK && board.fullMoveNumber() <= 15) {
            int file = sq % 8;
            int rank = sq / 8;
            int home_rank = (p.color() == Color::WHITE) ? 0 : 7;
            if (file == 0 || file == 7) val -= 40; // Phạt ra rìa
            if (rank == home_rank) val -= 20; // Phạt nằm ở hàng cuối
        }

        if (board.fullMoveNumber() <= 10 && p.type() == PieceType::QUEEN) val -= 20;

        Square enemy_king = board.kingSq(enemy);
        if (enemy_king != Square::underlying::NO_SQ) {
            int dist = std::abs((s.index() % 8) - (enemy_king.index() % 8)) + 
                       std::abs((s.index() / 8) - (enemy_king.index() / 8));
            if (dist <= 2) val += 15; 
        }

        score += sign * val;
    }

    // 2. Cặp Tượng
    int w_bishops = 0, b_bishops = 0;
    for(int sq=0; sq<64; sq++) {
        Piece p = board.at(sq);
        if(p != Piece::NONE && p.type() == PieceType::BISHOP) {
            if(p.color()==Color::WHITE) w_bishops++; else b_bishops++;
        }
    }
    if (w_bishops >= 2) score += 35;
    if (b_bishops >= 2) score -= 35;

    // 3. Cấu trúc Tốt & Tốt thông
    for (int file = 0; file < 8; file++) {
        int w_pawns = 0, b_pawns = 0;
        for (int rank = 0; rank < 8; rank++) {
            Piece p = board.at(chess::Square(chess::File(file), chess::Rank(rank)));
            if (p != Piece::NONE && p.type() == PieceType::PAWN) {
                if (p.color() == Color::WHITE) w_pawns++;
                else b_pawns++;
            }
        }
        if (w_pawns > 1) score -= 20 * (w_pawns - 1);
        if (b_pawns > 1) score += 20 * (b_pawns - 1);
    }

    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.at(sq);
        if (p == Piece::NONE || p.type() != PieceType::PAWN) continue;

        int sign = (p.color() == Color::WHITE) ? 1 : -1;
        int file = sq % 8;
        int rank = sq / 8;
        int rel_rank = (p.color() == Color::WHITE) ? rank : (7 - rank);
        
        bool is_passed = true;
        for (int df = -1; df <= 1; df++) {
            int nf = file + df;
            if (nf < 0 || nf >= 8) continue;
            for (int r = rank + (p.color() == Color::WHITE ? 1 : -1); r >= 0 && r < 8; r += (p.color() == Color::WHITE ? 1 : -1)) {
                Piece enemy_p = board.at(chess::Square(chess::File(nf), chess::Rank(r)));
                if (enemy_p != Piece::NONE && enemy_p.type() == PieceType::PAWN && enemy_p.color() != p.color()) {
                    is_passed = false;
                    break;
                }
            }
            if (!is_passed) break;
        }
        
        if (is_passed) {
            if (rel_rank >= 2) score += sign * (rel_rank * 15);
            if (rel_rank == 6) score += sign * 50;
        }
    }

    // 4. Xe trên cột mở
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.at(sq);
        if (p == Piece::NONE || p.type() != PieceType::ROOK) continue;

        int sign = (p.color() == Color::WHITE) ? 1 : -1;
        int file = sq % 8;
        int rank = sq / 8;
        
        bool own_pawn_on_file = false;
        bool enemy_pawn_on_file = false;
        for (int r = 0; r < 8; r++) {
            Piece pawn = board.at(chess::Square(chess::File(file), chess::Rank(r)));
            if (pawn != Piece::NONE && pawn.type() == PieceType::PAWN) {
                if (pawn.color() == p.color()) own_pawn_on_file = true;
                else enemy_pawn_on_file = true;
            }
        }
        
        if (!own_pawn_on_file && !enemy_pawn_on_file) score += sign * 25;
        else if (!own_pawn_on_file && enemy_pawn_on_file) score += sign * 15;
        
        int rel_rank = (p.color() == Color::WHITE) ? rank : (7 - rank);
        if (rel_rank == 6) score += sign * 30;
    }

    // 5. Mã tiền đồn
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.at(sq);
        if (p == Piece::NONE || p.type() != PieceType::KNIGHT) continue;

        int sign = (p.color() == Color::WHITE) ? 1 : -1;
        int rank = sq / 8;
        int rel_rank = (p.color() == Color::WHITE) ? rank : (7 - rank);

        if (rel_rank >= 2 && rel_rank <= 5) {
            Square s = static_cast<Square>(sq);
            if (board.isAttacked(s, p.color())) score += sign * 20;
        }
    }

    // 6. An toàn Vua & NHẬP THÀNH SỚM
    for (Color color : {Color::WHITE, Color::BLACK}) {
        int sign = (color == Color::WHITE) ? 1 : -1;
        Square k_sq = board.kingSq(color);
        if (k_sq == Square::underlying::NO_SQ) continue;
        
        int kf = k_sq.index() % 8;
        int kr = k_sq.index() / 8;
        int home_rank = (color == Color::WHITE) ? 0 : 7;
        bool castled = ((color == Color::WHITE && kr == 0 && (kf == 2 || kf == 6)) ||
                        (color == Color::BLACK && kr == 7 && (kf == 2 || kf == 6)));

        // Thưởng lớn nếu đã nhập thành
        if (castled) {
            score += sign * 100; 
            int direction = (color == Color::WHITE) ? 1 : -1;
            int shield = 0;
            for (int df = -1; df <= 1; df++) {
                int nf = kf + df, nr = kr + direction;
                if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                    Piece p = board.at(chess::Square(chess::File(nf), chess::Rank(nr)));
                    if (p != Piece::NONE && p.color() == color && p.type() == PieceType::PAWN) shield++;
                }
            }
            if (shield == 0) score -= sign * 60;
            else if (shield == 1) score -= sign * 30;
        } else {
            // PHẠT NẶNG NẾU KHÔNG NHẬP THÀNH SAU NƯỚC 10
            if (board.fullMoveNumber() > 10) score -= sign * 60;
            // Phạt Vua đi lang thang
            if (!endgame && board.fullMoveNumber() <= 20 && kr != home_rank) {
                score -= sign * 55; 
            }
        }
    }

    // 7. ENDGAME MATE DRIVE (Dắt Vua tàn cuộc)
    if (endgame) {
        if (score > 300) {
            Square bk = board.kingSq(Color::BLACK);
            if (bk != Square::underlying::NO_SQ) {
                int bf = bk.index() % 8;
                int br = bk.index() / 8;
                int edge_dist = std::min(std::min(bf, 7 - bf), std::min(br, 7 - br));
                score += (7 - edge_dist) * 20; 
                
                Square wk = board.kingSq(Color::WHITE);
                if (wk != Square::underlying::NO_SQ) {
                    int dist = std::abs((wk.index() % 8) - bf) + std::abs((wk.index() / 8) - br);
                    score += (14 - dist) * 10; 
                }
            }
        } else if (score < -300) {
            Square wk = board.kingSq(Color::WHITE);
            if (wk != Square::underlying::NO_SQ) {
                int wf = wk.index() % 8;
                int wr = wk.index() / 8;
                int edge_dist = std::min(std::min(wf, 7 - wf), std::min(wr, 7 - wr));
                score -= (7 - edge_dist) * 20; 
                
                Square bk = board.kingSq(Color::BLACK);
                if (bk != Square::underlying::NO_SQ) {
                    int dist = std::abs((wk.index() % 8) - (bk.index() % 8)) + std::abs((wk.index() / 8) - (bk.index() / 8));
                    score -= (14 - dist) * 10;
                }
            }
        }
    }

    return score;
}

int evaluate_board(Board& board) {
    if (board.isGameOver().second == GameResult::LOSE) return -MATE_SCORE;
    if (board.isGameOver().second == GameResult::DRAW) return 0;

    int score = evaluate_white_score(board);
    int current_score = (board.sideToMove() == Color::WHITE) ? score : -score;

    if (board.isRepetition(2)) {
        if (current_score > 100) return -500; 
        if (current_score < -100) return 50;
        return 0;
    }
    return current_score;
}

void order_moves(Board& board, Movelist& moves) {
    struct SM { Move m; int s; };
    std::vector<SM> sm;
    sm.reserve(moves.size());
    
    for (int i = 0; i < moves.size(); i++) {
        Move move = moves[i];
        int score = 0;
        
        if (board.isCapture(move)) {
            Piece victim = board.at(move.to());
            if (victim != Piece::NONE) score += 10000 + 10 * PIECE_VALUES[static_cast<int>(victim.type())] - PIECE_VALUES[static_cast<int>(board.at(move.from()).type())];
            else score += 10000; 
        }
        
        if (move.promotionType() != PieceType::NONE) score += 9000;
        if (static_cast<int>(board.givesCheck(move)) != 0) score += 5000;

        sm.push_back({move, score});
    }
    std::sort(sm.begin(), sm.end(), [](const SM& a, const SM& b) { return a.s > b.s; });
    for (int i = 0; i < moves.size(); i++) moves[i] = sm[i].m;
}

int quiescence(Board& board, int alpha, int beta, int depth) {
    if (time_is_up) return evaluate_board(board);
    
    total_nodes++;
    if ((total_nodes & 1023) == 0) { 
        auto now = std::chrono::steady_clock::now(); 
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= TIME_LIMIT_MS) {
            time_is_up = true;
            return evaluate_board(board);
        }
    }

    bool in_check = board.inCheck();
    
    if (!in_check) {
        int stand_pat = evaluate_board(board);
        if (stand_pat >= beta) return beta;
        if (alpha < stand_pat) alpha = stand_pat;
    }

    if (depth == 0) return evaluate_board(board);

    Movelist moves; movegen::legalmoves(moves, board);
    if (moves.size() == 0) return in_check ? -MATE_SCORE : 0;

    Movelist search_moves;
    for(int i=0; i<moves.size(); i++) {
        if (in_check || board.isCapture(moves[i]) || static_cast<int>(board.givesCheck(moves[i])) != 0) {
            search_moves.add(moves[i]);
        }
    }
    
    order_moves(board, search_moves);

    for (int i = 0; i < search_moves.size(); i++) {
        board.makeMove(search_moves[i]);
        int score = -quiescence(board, -beta, -alpha, depth - 1);
        board.unmakeMove(search_moves[i]);
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

int negamax(Board& board, int depth, int alpha, int beta) {
    if (time_is_up) return evaluate_board(board);

    total_nodes++;
    if ((total_nodes & 1023) == 0) { 
        auto now = std::chrono::steady_clock::now(); 
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= TIME_LIMIT_MS) {
            time_is_up = true;
            return evaluate_board(board);
        }
    }

    if (depth == 0) return quiescence(board, alpha, beta, MAX_Q_DEPTH);

    bool in_check = board.inCheck();
    if (transposition_table.size() > 200000) transposition_table.clear();

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
    
    order_moves(board, moves);
    
    int best = -INF;
    Move best_move_in_node = moves[0];
    TTFlag flag = UPPERBOUND;

    for (int i = 0; i < moves.size(); i++) {
        bool is_capture = board.isCapture(moves[i]);
        bool gives_check = static_cast<int>(board.givesCheck(moves[i])) != 0;

        board.makeMove(moves[i]);
        
        int extension = board.inCheck() ? 1 : 0;
        int new_depth = depth - 1 + extension;

        if (i >= 3 && depth >= 3 && !in_check && !is_capture && !gives_check) {
            new_depth -= 1; 
        }

        int score;
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

    transposition_table[hash] = {depth, best, flag, best_move_in_node};
    return best;
}

std::string extract_fen(const std::string& body) {
    size_t start = body.find("\"fen\":\"");
    if (start == std::string::npos) {
        start = body.find("\"fen\": \"");
        if (start == std::string::npos) return "";
        start += 8; 
    } else {
        start += 7;
    }
    size_t end = body.find("\"", start);
    if (end == std::string::npos) return "";
    return body.substr(start, end - start);
}

int main() {
    init_zobrist();
    std::cout << "Khởi động Nyra Engine (Bản Vị Trí + Mở Rộng Khai Cuộc)..." << std::endl;

    httplib::Server svr;

    svr.Options(R"(/.*)", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 200;
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content("{\"status\":\"ready\"}", "application/json");
    });

    svr.Post("/move", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        try {
            std::string fen = extract_fen(req.body);
            if (fen.empty()) {
                res.status = 400; res.set_content("{\"error\":\"No FEN\"}", "application/json"); return;
            }

            std::string fen_short = fen.substr(0, fen.find_last_of(" "));
            fen_short = fen_short.substr(0, fen_short.find_last_of(" ")); 

            Board board = Board(fen);
            Movelist moves; 
            movegen::legalmoves(moves, board);
            
            if (moves.size() == 0) {
                res.set_content("{\"uci\":\"NONE\"}", "application/json"); return;
            }

            if (opening_book.find(fen_short) != opening_book.end()) {
                std::string book_move_uci = opening_book[fen_short];
                for (int i = 0; i < moves.size(); i++) {
                    if (uci::moveToUci(moves[i]) == book_move_uci) {
                        res.set_content("{\"uci\":\"" + book_move_uci + "\", \"search\":{\"depth\":0, \"method\":\"Opening Book\"}}", "application/json");
                        return;
                    }
                }
            }

            transposition_table.clear();
            transposition_table.reserve(200000); 
            order_moves(board, moves);
            
            int best_score = -INF;
            Move best_move = moves[0];
            int completed_depth = 0;

            time_is_up = false;
            total_nodes = 0;
            start_time = std::chrono::steady_clock::now();

            for (int depth = 1; depth <= MAX_DEPTH; depth++) {
                if (time_is_up) break;

                int current_best_score = -INF;
                Move current_best_move = moves[0];
                
                int alpha = -INF;
                int beta = INF;

                for (int i = 0; i < moves.size(); i++) {
                    board.makeMove(moves[i]);
                    int score = -negamax(board, depth - 1, -beta, -alpha);
                    board.unmakeMove(moves[i]);

                    if (time_is_up) break;

                    if (score > current_best_score) {
                        current_best_score = score;
                        current_best_move = moves[i];
                    }
                    if (score > alpha) alpha = score;
                }

                if (!time_is_up) {
                    best_move = current_best_move;
                    best_score = current_best_score;
                    completed_depth = depth;
                }
            }

            int ms = (int)std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
            
            std::string uci = uci::moveToUci(best_move);
            std::string json_resp = "{\"uci\":\"" + uci + "\", \"search\":{\"depth\":" + std::to_string(completed_depth) + ", \"time_ms\":" + std::to_string(ms) + ", \"nodes\":" + std::to_string(total_nodes) + "}}";
            res.set_content(json_resp, "application/json");
            
        } catch (const std::exception& e) {
            std::cerr << "Lỗi C++: " << e.what() << std::endl;
            res.status = 500; res.set_content("{\"error\":\"Internal Engine Error\"}", "application/json");
        }
    });

    int port = 8081;
    if (const char* env_p = std::getenv("PORT")) port = std::stoi(env_p);
    
    std::cout << "Server đang chạy ở cổng " << port << "..." << std::endl;
    svr.listen("0.0.0.0", port);
    return 0;
}
