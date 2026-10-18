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

// --- OPENING BOOK (GỌN: King's Pawn, Queen's Gambit, Italian, Caro-Kann, Sicilian) ---
// Mỗi FEN rút gọn có thể có nhiều nước chính thống, chọn ngẫu nhiên 1.
std::unordered_map<std::string, std::vector<std::string>> opening_book = {
    // 1.e4 → King's Pawn (e5) / Sicilian (c5) / Caro-Kann (c6)
    {"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -",
        {"e7e5", "c7c5", "c7c6"}},

    // 1.d4 → Queen's Pawn (d5)
    {"rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -",
        {"d7d5"}},

    // 1.e4 e5 (lượt trắng) → 2.Nf3
    {"rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
        {"g1f3"}},

    // 1.e4 e5 2.Nf3 Nc6 (lượt trắng) → 3.Bc4 (Italian Game)
    {"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -",
        {"f1c4"}},

    // 1.e4 c5 (lượt trắng) → 2.Nf3 (Sicilian)
    {"rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
        {"g1f3"}},

    // 1.e4 c5 2.Nf3 d6 (lượt trắng) → 3.d4 (Sicilian main line)
    {"rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -",
        {"d2d4"}},

    // 1.e4 c6 (lượt trắng) → 2.d4 (Caro-Kann)
    {"rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
        {"d2d4"}},

    // 1.e4 c6 2.d4 d5 (lượt trắng) → 3.Nc3 (Caro-Kann Classical)
    {"rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq -",
        {"b1c3"}},

    // 1.d4 d5 (lượt trắng) → 2.c4 (Queen's Gambit)
    {"rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -",
        {"c2c4"}},

    // 1.d4 d5 2.c4 (lượt đen) → 2...e6 (QGD) hoặc 2...c6 (Slav)
    {"rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq -",
        {"e7e6", "c7c6"}},
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
        if (is_attacked && !is_defended && p.type() != PieceType::KING) {
            val -= static_cast<int>(PIECE_VALUES[pt_idx] * 0.9);
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

    // 6. An toàn Vua
    for (Color color : {Color::WHITE, Color::BLACK}) {
        int sign = (color == Color::WHITE) ? 1 : -1;
        Square k_sq = board.kingSq(color);
        if (k_sq == Square::underlying::NO_SQ) continue;
        
        int kf = k_sq.index() % 8;
        int kr = k_sq.index() / 8;
        int home_rank = (color == Color::WHITE) ? 0 : 7;
        bool castled = ((color == Color::WHITE && kr == 0 && (kf == 2 || kf == 6)) ||
                        (color == Color::BLACK && kr == 7 && (kf == 2 || kf == 6)));

        if (castled) {
            score += sign * 70;
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
        }

        if (!endgame && board.fullMoveNumber() <= 20 && kr != home_rank && !castled) {
            score -= sign * 45; 
        }
    }

    // 7. ENDGAME MATE DRIVE
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

// --- KIỂM TRA NƯỚC XE VÔ NGHĨA TRONG 10 NƯỚC ĐẦU ---
// Trả về true nếu là nước Xe KHÔNG có mục đích (không ăn quân, không chiếm cột mở,
// không tấn công quân giá trị >= 300, không tiến đến hàng 6/7/8 tính từ phía mình).
// Nhập thành không bị ảnh hưởng vì moving.type() = KING, không phải ROOK.
bool is_rook_move_wasteful(Board& board, Move move) {
    // Chỉ xét trong 10 nước đầu
    if (board.fullMoveNumber() > 10) return false;

    Piece moving = board.at(move.from());
    if (moving == Piece::NONE || moving.type() != PieceType::ROOK) return false;

    // Ăn quân → có lợi
    if (board.isCapture(move)) return false;

    Color us = moving.color();
    int to_file = move.to().index() % 8;
    int to_rank = move.to().index() / 8;

    board.makeMove(move);
    bool useful = false;

    // 1. Xe chiếm cột mở hoặc bán mở (không còn tốt mình trên cột)
    bool own_pawn_on_file = false;
    for (int r = 0; r < 8; r++) {
        Piece p = board.at(Square(File(to_file), Rank(r)));
        if (p != Piece::NONE && p.type() == PieceType::PAWN && p.color() == us) {
            own_pawn_on_file = true;
            break;
        }
    }
    if (!own_pawn_on_file) useful = true;

    // 2. Xe tấn công quân địch giá trị >= 300 (Tượng, Mã, Xe, Hậu)
    for (int dir = 0; dir < 4 && !useful; dir++) {
        int df = (dir == 0) ? 1 : (dir == 1) ? -1 : 0;
        int dr = (dir == 2) ? 1 : (dir == 3) ? -1 : 0;
        int f = to_file + df, r = to_rank + dr;
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            Piece p = board.at(Square(File(f), Rank(r)));
            if (p != Piece::NONE) {
                if (p.color() != us && PIECE_VALUES[static_cast<int>(p.type())] >= 300) {
                    useful = true;
                }
                break;
            }
            f += df; r += dr;
        }
    }

    // 3. Xe tiến đến hàng 6/7/8 tính từ phía mình
    int rel_rank = (us == Color::WHITE) ? to_rank : (7 - to_rank);
    if (rel_rank >= 6) useful = true;

    board.unmakeMove(move);
    return !useful;
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

        // Phạt nặng nước Xe vô nghĩa trong 10 nước đầu (đẩy xuống cuối)
        if (is_rook_move_wasteful(board, move)) score -= 50000;

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

        // Đánh dấu nước Xe vô nghĩa để phạt nhẹ sau khi tính điểm
        bool wasteful_rook = is_rook_move_wasteful(board, moves[i]);

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

        // Phạt nhẹ nước Xe vô nghĩa — đủ để engine tránh nếu có lựa chọn tương đương
        if (wasteful_rook) score -= 200;
        
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
    std::cout << "Khởi động Nyra Engine (Giai đoạn 3: Book + Endgame)..." << std::endl;

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

            // Lấy FEN rút gọn để so sánh với Opening Book
            std::string fen_short = fen.substr(0, fen.find_last_of(" "));
            fen_short = fen_short.substr(0, fen_short.find_last_of(" ")); 

            Board board = Board(fen);
            Movelist moves; 
            movegen::legalmoves(moves, board);
            
            if (moves.size() == 0) {
                res.set_content("{\"uci\":\"NONE\"}", "application/json"); return;
            }

            // 1. Kiểm tra Opening Book trước (chọn ngẫu nhiên trong tập nước chính thống)
            auto book_it = opening_book.find(fen_short);
            if (book_it != opening_book.end() && !book_it->second.empty()) {
                const std::vector<std::string>& candidates = book_it->second;
                static std::mt19937 book_rng(
                    static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count())
                );
                std::uniform_int_distribution<size_t> pick(0, candidates.size() - 1);
                std::string book_move_uci = candidates[pick(book_rng)];

                for (int i = 0; i < moves.size(); i++) {
                    if (uci::moveToUci(moves[i]) == book_move_uci) {
                        res.set_content("{\"uci\":\"" + book_move_uci + "\", \"search\":{\"depth\":0, \"method\":\"Opening Book\"}}", "application/json");
                        return;
                    }
                }
                // Nếu nước book không hợp lệ → fallback sang search
            }

            // 2. Nếu không có trong sách, chạy thuật toán Search
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
