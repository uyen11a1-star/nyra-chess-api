#include "chess.hpp"
#include "httplib.h"
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <sstream>
#include <unordered_map>
#include <random>
#include <mutex>
#include <thread>
#include <atomic>

using namespace chess;
typedef uint64_t U64;

// ================= CẤU HÌNH =================
int TIME_LIMIT_MS = 2000;  // tối đa mỗi nước (mili-giây, giới hạn cứng)
int SOFT_LIMIT_MS = 1300;  // quá mốc này thì không bắt đầu độ sâu mới
const int MAX_DEPTH = 64;
const int MAX_PLY = 96;
const int MATE_SCORE = 100000;
const int MATE_BOUND = MATE_SCORE - 1000;
const int INF = 200000;
const int TT_BITS = 20;  // 2^20 ô (~24MB)

// Chỉ số quân theo chess.hpp: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
const int VAL[6] = {100, 320, 330, 500, 900, 20000};

// ================= PST (hàng 8 nằm trên cùng, trắng dùng sq^56, đen dùng sq) =================
const int PAWN_MG[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0};
const int PAWN_EG[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    70, 70, 70, 70, 70, 70, 70, 70,
    40, 40, 40, 40, 40, 40, 40, 40,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
     5,  5,  5,  5,  5,  5,  5,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0};
const int KNIGHT_PST[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50};
const int BISHOP_PST[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20};
const int ROOK_PST[64] = {
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0};
const int QUEEN_PST[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20};
const int KING_MG[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20};
const int KING_EG[64] = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50};
const int* PST_MG[6] = {PAWN_MG, KNIGHT_PST, BISHOP_PST, ROOK_PST, QUEEN_PST, KING_MG};
const int* PST_EG[6] = {PAWN_EG, KNIGHT_PST, BISHOP_PST, ROOK_PST, QUEEN_PST, KING_EG};
const PieceType PTS[6] = {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                          PieceType::ROOK, PieceType::QUEEN, PieceType::KING};

const int PASSED_MG[8] = {0, 0, 5, 10, 20, 40, 70, 0};
const int PASSED_EG[8] = {0, 5, 10, 25, 45, 80, 130, 0};
const int ATK_W[6] = {0, 2, 2, 3, 5, 0};

// ================= MẶT NẠ BITBOARD =================
U64 FILE_MASK[8], RANK_MASK[8], ADJ_FILES[8];
U64 PASSED_MASK[2][64], ADJ_FRONT[2][64];

void init_masks() {
    for (int f = 0; f < 8; f++) FILE_MASK[f] = 0x0101010101010101ULL << f;
    for (int r = 0; r < 8; r++) RANK_MASK[r] = 0xFFULL << (8 * r);
    for (int f = 0; f < 8; f++)
        ADJ_FILES[f] = (f > 0 ? FILE_MASK[f - 1] : 0) | (f < 7 ? FILE_MASK[f + 1] : 0);
    for (int sq = 0; sq < 64; sq++) {
        int f = sq % 8, r = sq / 8;
        U64 front_w = 0, front_b = 0;
        for (int rr = r + 1; rr < 8; rr++) front_w |= RANK_MASK[rr];
        for (int rr = 0; rr < r; rr++) front_b |= RANK_MASK[rr];
        PASSED_MASK[0][sq] = front_w & (FILE_MASK[f] | ADJ_FILES[f]);
        PASSED_MASK[1][sq] = front_b & (FILE_MASK[f] | ADJ_FILES[f]);
        ADJ_FRONT[0][sq] = front_w & ADJ_FILES[f];
        ADJ_FRONT[1][sq] = front_b & ADJ_FILES[f];
    }
}

inline int popcnt(U64 b) { return __builtin_popcountll(b); }
inline int lsb_idx(U64 b) { return __builtin_ctzll(b); }
inline int cheb(int a, int b) { return std::max(std::abs(a % 8 - b % 8), std::abs(a / 8 - b / 8)); }
inline int manh(int a, int b) { return std::abs(a % 8 - b % 8) + std::abs(a / 8 - b / 8); }

// ================= ĐÁNH GIÁ (điểm theo phía đang đi) =================
int evaluate(const Board& board) {
    int mg[2] = {0, 0}, eg[2] = {0, 0};
    int phase = 0, npm[2] = {0, 0}, pc[2] = {0, 0}, bishops[2] = {0, 0};
    int atk_units[2] = {0, 0}, atk_cnt[2] = {0, 0};
    U64 pawns[2], pawn_att[2], kz[2];
    int ksq[2];
    const Bitboard occ = board.occ();

    pawns[0] = board.pieces(PieceType::PAWN, Color::WHITE).getBits();
    pawns[1] = board.pieces(PieceType::PAWN, Color::BLACK).getBits();
    pawn_att[0] = ((pawns[0] & ~FILE_MASK[0]) << 7) | ((pawns[0] & ~FILE_MASK[7]) << 9);
    pawn_att[1] = ((pawns[1] & ~FILE_MASK[0]) >> 9) | ((pawns[1] & ~FILE_MASK[7]) >> 7);
    for (int c = 0; c < 2; c++) {
        Color col = c == 0 ? Color::WHITE : Color::BLACK;
        ksq[c] = board.kingSq(col).index();
        kz[c] = attacks::king(Square(ksq[c])).getBits() | (1ULL << ksq[c]);
        pc[c] = popcnt(pawns[c]);
    }

    for (int c = 0; c < 2; c++) {
        const int e = c ^ 1;
        const Color col = c == 0 ? Color::WHITE : Color::BLACK;
        const U64 own = board.us(col).getBits();

        for (int pt = 0; pt < 6; pt++) {
            U64 bb = board.pieces(PTS[pt], col).getBits();
            while (bb) {
                int sq = lsb_idx(bb);
                bb &= bb - 1;
                int idx = (c == 0) ? (sq ^ 56) : sq;
                mg[c] += VAL[pt] + PST_MG[pt][idx];
                eg[c] += VAL[pt] + PST_EG[pt][idx];
                int rel_rank = (c == 0) ? sq / 8 : 7 - sq / 8;
                int file = sq % 8;

                if (pt == 0) {  // TỐT
                    if (!(PASSED_MASK[c][sq] & pawns[e])) {
                        mg[c] += PASSED_MG[rel_rank];
                        eg[c] += PASSED_EG[rel_rank];
                        int stop = (c == 0) ? sq + 8 : sq - 8;
                        if (stop >= 0 && stop < 64) {
                            eg[c] += (cheb(ksq[e], stop) - cheb(ksq[c], stop)) * rel_rank;
                        }
                    }
                    if (!(ADJ_FILES[file] & pawns[c])) { mg[c] -= 12; eg[c] -= 15; }  // tốt cô lập
                    if ((pawn_att[c] >> sq) & 1) { mg[c] += 6; eg[c] += 4; }          // tốt được tốt bảo vệ
                    continue;
                }
                if (pt == 5) continue;  // VUA xử lý riêng

                U64 att = 0;
                if (pt == 1) {
                    npm[c] += VAL[1]; phase += 1;
                    att = attacks::knight(Square(sq)).getBits();
                    int mob = popcnt(att & ~own & ~pawn_att[e]);
                    mg[c] += (mob - 4) * 4; eg[c] += (mob - 4) * 4;
                    if (rel_rank >= 3 && rel_rank <= 5 && !(ADJ_FRONT[c][sq] & pawns[e]) &&
                        ((pawn_att[c] >> sq) & 1)) { mg[c] += 20; eg[c] += 10; }  // mã tiền đồn
                } else if (pt == 2) {
                    npm[c] += VAL[2]; phase += 1; bishops[c]++;
                    att = attacks::bishop(Square(sq), occ).getBits();
                    int mob = popcnt(att & ~own & ~pawn_att[e]);
                    mg[c] += (mob - 6) * 4; eg[c] += (mob - 6) * 5;
                } else if (pt == 3) {
                    npm[c] += VAL[3]; phase += 2;
                    att = attacks::rook(Square(sq), occ).getBits();
                    int mob = popcnt(att & ~own);
                    mg[c] += (mob - 7) * 2; eg[c] += (mob - 7) * 4;
                    if (!(pawns[c] & FILE_MASK[file])) {
                        if (!(pawns[e] & FILE_MASK[file])) { mg[c] += 25; eg[c] += 15; }
                        else { mg[c] += 12; eg[c] += 8; }
                    }
                    if (rel_rank == 6) { mg[c] += 15; eg[c] += 25; }
                } else {
                    npm[c] += VAL[4]; phase += 4;
                    att = attacks::queen(Square(sq), occ).getBits();
                    int mob = popcnt(att & ~own);
                    mg[c] += (mob - 14); eg[c] += (mob - 14) * 2;
                }
                if (att & kz[e]) { atk_cnt[c]++; atk_units[c] += ATK_W[pt]; }
            }
        }

        if (bishops[c] >= 2) { mg[c] += 30; eg[c] += 45; }

        // Vua: tốt che chắn, cột mở quanh vua
        int kf = ksq[c] % 8, kr = ksq[c] / 8;
        int rel_k = (c == 0) ? kr : 7 - kr;
        U64 tri = (FILE_MASK[kf] | ADJ_FILES[kf]) & RANK_MASK[kr];
        U64 shield_sq = (c == 0) ? ((tri << 8) | (tri << 16)) : ((tri >> 8) | (tri >> 16));
        int shield = popcnt(shield_sq & pawns[c]);
        if (rel_k <= 1) mg[c] += std::min(shield, 4) * 8 - 20;
        for (int ff = std::max(0, kf - 1); ff <= std::min(7, kf + 1); ff++) {
            if (!(pawns[c] & FILE_MASK[ff])) mg[c] -= (pawns[e] & FILE_MASK[ff]) ? 10 : 18;
        }
    }
    // nguy hiểm khi bị nhiều quân tấn công vùng vua
    for (int c = 0; c < 2; c++) {
        int e = c ^ 1;
        if (atk_cnt[e] >= 2) mg[c] -= std::min(atk_units[e] * atk_units[e] / 2, 300);
    }

    int p = std::min(phase, 24);
    int score = ((mg[0] - mg[1]) * p + (eg[0] - eg[1]) * (24 - p)) / 24;

    int diff = (npm[0] + 100 * pc[0]) - (npm[1] + 100 * pc[1]);
    int winner = score > 0 ? 0 : 1;
    int loser = winner ^ 1;

    // Tàn cuộc: dồn vua yếu ra biên, đưa vua mạnh lại gần (liên tục theo chênh lệch vật chất)
    int ad = std::abs(diff);
    if (phase <= 12 && ad >= 300 && ((diff > 0) == (winner == 0))) {
        int lk = ksq[loser];
        int cmd = std::max(3 - lk % 8, lk % 8 - 4) + std::max(3 - lk / 8, lk / 8 - 4);
        int bonus = (cmd * 10 + (14 - manh(ksq[0], ksq[1])) * 4) * std::min(ad - 200, 600) / 600;
        score += (winner == 0) ? bonus : -bonus;
    }
    // Không còn tốt và chỉ hơn tối đa một quân nhẹ: gần như hòa
    if (pc[winner] == 0 && (npm[winner] - npm[loser]) <= 330) score /= 4;

    int stm_score = (board.sideToMove() == Color::WHITE) ? score : -score;
    return stm_score + 10;
}

// ================= BẢNG BĂM =================
enum TTFlag : uint8_t { TT_EXACT, TT_LOWER, TT_UPPER };
struct TTEntry {
    U64 key = 0;
    int score = 0;
    Move move = Move::NO_MOVE;
    int8_t depth = 0;
    uint8_t flag = 0;
};
std::vector<TTEntry> tt;
const U64 TT_MASK = (1ULL << TT_BITS) - 1;

inline int to_tt(int s, int ply) { return s >= MATE_BOUND ? s + ply : (s <= -MATE_BOUND ? s - ply : s); }
inline int from_tt(int s, int ply) { return s >= MATE_BOUND ? s - ply : (s <= -MATE_BOUND ? s + ply : s); }

// ================= TRẠNG THÁI TÌM KIẾM =================
std::chrono::steady_clock::time_point start_time;
std::atomic<bool> stop_search(false);
long long search_limit_ms = 2000;
long long total_nodes = 0;
Move killers[MAX_PLY + 2][2];
int history_tbl[2][64][64];
int lmr_table[64][64];
Move root_best = Move::NO_MOVE;
Move root_pv_move = Move::NO_MOVE;
std::mutex engine_mutex;

inline long long elapsed_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
}

void init_search_tables() {
    for (int d = 1; d < 64; d++)
        for (int m = 1; m < 64; m++)
            lmr_table[d][m] = (int)(0.75 + std::log((double)d) * std::log((double)m) / 2.25);
}

inline bool has_non_pawn(const Board& b, Color c) {
    return !(b.pieces(PieceType::KNIGHT, c) | b.pieces(PieceType::BISHOP, c) |
             b.pieces(PieceType::ROOK, c) | b.pieces(PieceType::QUEEN, c)).empty();
}

inline int victim_idx(const Board& b, const Move& m) {
    return m.typeOf() == Move::ENPASSANT ? 0 : (int)b.at(m.to()).type();
}

void update_history(int stm, const Move& m, int bonus) {
    int& h = history_tbl[stm][m.from().index()][m.to().index()];
    h += bonus - h * std::abs(bonus) / 16384;
}

// ================= QUIESCENCE (chỉ bắt quân, hoặc né chiếu) =================
int quiescence(Board& board, int alpha, int beta, int ply) {
    if (stop_search) return 0;
    total_nodes++;
    if ((total_nodes & 2047) == 0 && elapsed_ms() >= search_limit_ms) { stop_search = true; return 0; }
    if (ply >= MAX_PLY - 1) return evaluate(board);

    bool in_check = board.inCheck();
    int best, stand = 0;
    if (in_check) {
        best = -MATE_SCORE + ply;
    } else {
        stand = evaluate(board);
        if (stand >= beta) return stand;
        if (stand > alpha) alpha = stand;
        best = stand;
    }

    Movelist moves;
    if (in_check) movegen::legalmoves(moves, board);
    else movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, board);

    int sc[256];
    Color opp = ~board.sideToMove();
    for (int i = 0; i < moves.size(); i++) {
        const Move& m = moves[i];
        int v = victim_idx(board, m);
        int a = (int)board.at(m.from()).type();
        sc[i] = (board.isCapture(m) ? 10 * VAL[v] - VAL[a] : -VAL[a]);
        if (m.typeOf() == Move::PROMOTION) sc[i] += 8000;
    }

    for (int i = 0; i < moves.size(); i++) {
        int bi = i;
        for (int j = i + 1; j < moves.size(); j++) if (sc[j] > sc[bi]) bi = j;
        std::swap(sc[i], sc[bi]); std::swap(moves[i], moves[bi]);
        const Move m = moves[i];

        if (!in_check) {
            bool promo = m.typeOf() == Move::PROMOTION;
            int v = victim_idx(board, m);
            int a = (int)board.at(m.from()).type();
            if (!promo && stand + VAL[v] + 200 < alpha) continue;            // delta pruning
            if (!promo && VAL[v] < VAL[a] && board.isAttacked(m.to(), opp)) continue;  // bắt lỗ vốn
        }

        board.makeMove(m);
        int score = -quiescence(board, -beta, -alpha, ply + 1);
        board.unmakeMove(m);
        if (stop_search) return 0;

        if (score > best) {
            best = score;
            if (score > alpha) alpha = score;
        }
        if (alpha >= beta) break;
    }
    return best;
}

// ================= NEGAMAX (PVS + TT + null move + LMR + pruning) =================
int negamax(Board& board, int depth, int alpha, int beta, int ply, bool allow_null) {
    if (stop_search) return 0;
    total_nodes++;
    if ((total_nodes & 2047) == 0 && elapsed_ms() >= search_limit_ms) { stop_search = true; return 0; }

    const bool pv = (beta - alpha) > 1;
    if (ply > 0) {
        if (board.isHalfMoveDraw() || board.isInsufficientMaterial() || board.isRepetition(1)) return 0;
        alpha = std::max(alpha, -MATE_SCORE + ply);
        beta = std::min(beta, MATE_SCORE - ply - 1);
        if (alpha >= beta) return alpha;
    }
    if (ply >= MAX_PLY - 1) return evaluate(board);

    const bool in_check = board.inCheck();
    if (in_check) depth++;
    if (depth <= 0) return quiescence(board, alpha, beta, ply);

    const Color stm_c = board.sideToMove();
    const int stm = stm_c == Color::WHITE ? 0 : 1;

    // --- Bảng băm ---
    const U64 key = board.hash();
    TTEntry& e = tt[key & TT_MASK];
    Move tt_move = Move::NO_MOVE;
    bool tt_hit = (e.key == key);
    if (tt_hit) {
        tt_move = e.move;
        if (ply > 0 && !pv && e.depth >= depth) {
            int s = from_tt(e.score, ply);
            if (e.flag == TT_EXACT) return s;
            if (e.flag == TT_LOWER && s >= beta) return s;
            if (e.flag == TT_UPPER && s <= alpha) return s;
        }
    }

    int static_eval = in_check ? -INF : evaluate(board);

    // --- Cắt tỉa trước khi duyệt nước đi ---
    if (!pv && !in_check && ply > 0) {
        if (depth <= 6 && static_eval - 80 * depth >= beta && std::abs(beta) < MATE_BOUND)
            return static_eval;  // reverse futility
        if (allow_null && depth >= 3 && static_eval >= beta && has_non_pawn(board, stm_c)) {
            int R = 3 + depth / 4;
            board.makeNullMove();
            int s = -negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
            board.unmakeNullMove();
            if (stop_search) return 0;
            if (s >= beta) return s >= MATE_BOUND ? beta : s;
        }
    }

    Movelist moves;
    movegen::legalmoves(moves, board);
    if (moves.size() == 0) return in_check ? -MATE_SCORE + ply : 0;

    // --- Chấm điểm sắp xếp nước đi ---
    int sc[256];
    for (int i = 0; i < moves.size(); i++) {
        const Move& m = moves[i];
        int s;
        if (m == tt_move) s = 2000000;
        else if (ply == 0 && m == root_pv_move) s = 1900000;
        else if (board.isCapture(m)) {
            int v = victim_idx(board, m);
            int a = (int)board.at(m.from()).type();
            bool bad = VAL[v] < VAL[a] && board.isAttacked(m.to(), ~stm_c);
            s = (bad ? -200000 : 500000) + 10 * VAL[v] - VAL[a];
        } else if (m.typeOf() == Move::PROMOTION) {
            s = 600000 + (m.promotionType() == PieceType::QUEEN ? 1000 : 0);
        } else if (m == killers[ply][0]) s = 90000;
        else if (m == killers[ply][1]) s = 80000;
        else s = history_tbl[stm][m.from().index()][m.to().index()];
        sc[i] = s;
    }

    int best = -INF;
    Move best_move = Move::NO_MOVE;
    const int orig_alpha = alpha;
    int move_count = 0;
    Move quiets[64];
    int nq = 0;

    for (int i = 0; i < moves.size(); i++) {
        int bi = i;
        for (int j = i + 1; j < moves.size(); j++) if (sc[j] > sc[bi]) bi = j;
        std::swap(sc[i], sc[bi]); std::swap(moves[i], moves[bi]);
        const Move m = moves[i];

        const bool capture = board.isCapture(m);
        const bool promo = m.typeOf() == Move::PROMOTION;
        const bool quiet = !capture && !promo;
        const bool gives_check = static_cast<int>(board.givesCheck(m)) != 0;
        move_count++;

        // --- Cắt tỉa nước yên lặng muộn ---
        if (!pv && !in_check && quiet && !gives_check && ply > 0 && best > -MATE_BOUND && move_count > 1) {
            if (depth <= 3 && move_count > 4 + depth * depth) continue;           // late move pruning
            if (depth <= 4 && static_eval + 100 + 120 * depth <= alpha) continue; // futility
        }

        board.makeMove(m);
        int new_depth = depth - 1;
        int score;
        if (move_count == 1) {
            score = -negamax(board, new_depth, -beta, -alpha, ply + 1, true);
        } else {
            int r = 0;
            if (depth >= 3 && move_count >= 3 && quiet && !in_check && !gives_check) {
                r = lmr_table[std::min(depth, 63)][std::min(move_count, 63)];
                if (pv) r--;
                if (m == killers[ply][0] || m == killers[ply][1]) r--;
                r -= history_tbl[stm][m.from().index()][m.to().index()] / 6000;
                if (r < 0) r = 0;
                if (r > new_depth - 1) r = std::max(0, new_depth - 1);
            }
            score = -negamax(board, new_depth - r, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && r > 0) score = -negamax(board, new_depth, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && score < beta) score = -negamax(board, new_depth, -beta, -alpha, ply + 1, true);
        }
        board.unmakeMove(m);
        if (stop_search) return 0;

        if (score > best) {
            best = score;
            best_move = m;
            if (score > alpha) {
                alpha = score;
                if (ply == 0) root_best = m;
            }
        }
        if (alpha >= beta) {
            if (quiet) {
                if (!(killers[ply][0] == m)) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
                int bonus = std::min(depth * depth, 400);
                update_history(stm, m, bonus);
                for (int k = 0; k < nq; k++) update_history(stm, quiets[k], -bonus);
            }
            break;
        }
        if (quiet && nq < 64) quiets[nq++] = m;
    }

    if (!stop_search) {
        uint8_t flag = best >= beta ? TT_LOWER : (best > orig_alpha ? TT_EXACT : TT_UPPER);
        if (e.key != key || depth >= e.depth || flag == TT_EXACT) {
            e.key = key;
            e.score = to_tt(best, ply);
            e.move = best_move;
            e.depth = (int8_t)std::min(depth, 100);
            e.flag = flag;
        }
    }
    return best;
}

// ================= TÌM NƯỚC ĐI (lặp sâu dần + aspiration) =================
struct SearchResult { Move move; int score; int depth; };

SearchResult search_position(Board& board) {
    stop_search = false;
    search_limit_ms = TIME_LIMIT_MS;
    total_nodes = 0;
    start_time = std::chrono::steady_clock::now();
    std::memset(history_tbl, 0, sizeof(history_tbl));
    for (int i = 0; i < MAX_PLY + 2; i++) killers[i][0] = killers[i][1] = Move::NO_MOVE;
    root_pv_move = Move::NO_MOVE;

    Movelist root_moves;
    movegen::legalmoves(root_moves, board);
    Move best = root_moves[0];
    int best_score = 0, done_depth = 0;
    if (root_moves.size() == 1) return {best, 0, 0};

    for (int depth = 1; depth <= MAX_DEPTH; depth++) {
        int alpha = -INF, beta = INF, delta = 30, score;
        if (depth >= 5) { alpha = best_score - delta; beta = best_score + delta; }
        while (true) {
            root_best = Move::NO_MOVE;
            score = negamax(board, depth, alpha, beta, 0, false);
            if (stop_search) break;
            if (score <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = std::max(score - delta, -INF);
                delta *= 2;
            } else if (score >= beta) {
                beta = std::min(score + delta, INF);
                delta *= 2;
            } else break;
            if (delta > 400) { alpha = -INF; beta = INF; }
        }
        if (stop_search) { if (root_best != Move::NO_MOVE) best = root_best; break; }
        if (root_best != Move::NO_MOVE) { best = root_best; root_pv_move = best; }
        best_score = score;
        done_depth = depth;
        if (elapsed_ms() >= SOFT_LIMIT_MS) break;
        if (std::abs(score) >= MATE_BOUND && depth > MATE_SCORE - std::abs(score)) break;
    }
    return {best, best_score, done_depth};
}

// ================= PONDERING (tìm trước trong lúc người chơi nghĩ) =================
int PONDER_LIMIT_MS = 20000;  // thời gian tối đa tìm trước (mili-giây), 0 = tắt
std::thread ponder_thread;

void stop_ponder() {  // dừng luồng tìm trước (nếu có) và chờ nó kết thúc hẳn
    if (ponder_thread.joinable()) {
        stop_search = true;
        ponder_thread.join();
    }
}

void ponder_search(Board board) {
    stop_search = false;
    search_limit_ms = PONDER_LIMIT_MS;
    total_nodes = 0;
    start_time = std::chrono::steady_clock::now();
    std::memset(history_tbl, 0, sizeof(history_tbl));
    for (int i = 0; i < MAX_PLY + 2; i++) killers[i][0] = killers[i][1] = Move::NO_MOVE;
    for (int depth = 1; depth <= MAX_DEPTH; depth++) {
        int score = negamax(board, depth, -INF, INF, 0, false);
        if (stop_search) break;
        if (std::abs(score) >= MATE_BOUND) break;
    }
}

void start_ponder(const Board& b) {
    if (PONDER_LIMIT_MS <= 0) return;
    stop_ponder();
    ponder_thread = std::thread(ponder_search, b);
}

// ================= OPENING BOOK (xây từ các dòng khai cuộc, có biến thể ngẫu nhiên) =================
std::unordered_map<std::string, std::vector<std::string>> opening_book;
std::mt19937 book_rng(std::random_device{}());

std::string book_key(const std::string& fen) {  // bỏ ô bắt tốt qua đường + bộ đếm nước
    std::istringstream ss(fen);
    std::string a, b, c;
    ss >> a >> b >> c;
    return a + " " + b + " " + c;
}

void init_book() {
    const char* lines[] = {
        "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8",
        "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6 e1g1 e8g8",
        "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5 c2c3 d7d6 e1g1 e8g8",
        "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5",
        "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6",
        "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5 f8e7 e4e5 f6d7 g5e7 d8e7",
        "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6",
        "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 b8d7",
        "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5 g1f3 c7c5",
        "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5",
        "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5 e2e3 e7e6",
        "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6 e1g1 f8e7 d2d3 e8g8",
        "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5 f1g2 d5b6"};
    for (const char* line : lines) {
        Board b;
        std::istringstream ss(line);
        std::string mv;
        while (ss >> mv) {
            std::vector<std::string>& v = opening_book[book_key(b.getFen())];
            if (std::find(v.begin(), v.end(), mv) == v.end()) v.push_back(mv);
            b.makeMove(uci::uciToMove(b, mv));
        }
    }
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
    init_masks();
    init_search_tables();
    init_book();
    tt.assign((size_t)1 << TT_BITS, TTEntry());
    if (const char* t = std::getenv("NYRA_TIME_MS")) {  // tuỳ chọn: chỉnh thời gian nghĩ khi test
        TIME_LIMIT_MS = std::max(50, std::stoi(t));
        SOFT_LIMIT_MS = TIME_LIMIT_MS * 13 / 20;
    }
    if (const char* pm = std::getenv("NYRA_PONDER_MS")) PONDER_LIMIT_MS = std::max(0, std::stoi(pm));
    std::cout << "Khởi động Nyra Engine (PVS + tapered eval + book)..." << std::endl;

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
            std::lock_guard<std::mutex> lock(engine_mutex);
            stop_ponder();
            std::string fen = extract_fen(req.body);
            if (fen.empty()) {
                res.status = 400; res.set_content("{\"error\":\"No FEN\"}", "application/json"); return;
            }
            Board board = Board(fen);
            Movelist moves;
            movegen::legalmoves(moves, board);
            if (moves.size() == 0) {
                res.set_content("{\"uci\":\"NONE\"}", "application/json"); return;
            }

            // 1. Opening book (chọn ngẫu nhiên giữa các nước hợp lệ trong sách)
            auto it = opening_book.find(book_key(fen));
            if (it != opening_book.end()) {
                std::vector<std::string> legal_book;
                for (const std::string& bm : it->second)
                    for (int i = 0; i < moves.size(); i++)
                        if (uci::moveToUci(moves[i]) == bm) { legal_book.push_back(bm); break; }
                if (!legal_book.empty()) {
                    std::string pick = legal_book[book_rng() % legal_book.size()];
                    res.set_content("{\"uci\":\"" + pick + "\", \"search\":{\"depth\":0, \"method\":\"Opening Book\"}}", "application/json");
                    return;
                }
            }

            // 2. Tìm kiếm
            SearchResult r = search_position(board);
            int ms = (int)elapsed_ms();
            std::string json_resp = "{\"uci\":\"" + uci::moveToUci(r.move) + "\", \"search\":{\"depth\":" +
                std::to_string(r.depth) + ", \"time_ms\":" + std::to_string(ms) +
                ", \"nodes\":" + std::to_string(total_nodes) + ", \"score\":" + std::to_string(r.score) + "}}";
            res.set_content(json_resp, "application/json");
            Board after = board;
            after.makeMove(r.move);
            start_ponder(after);  // tìm trước trong lúc người chơi nghĩ
        } catch (const std::exception& ex) {
            std::cerr << "Lỗi C++: " << ex.what() << std::endl;
            res.status = 500; res.set_content("{\"error\":\"Internal Engine Error\"}", "application/json");
        }
    });

    int port = 8081;
    if (const char* env_p = std::getenv("PORT")) port = std::stoi(env_p);
    std::cout << "Server đang chạy ở cổng " << port << "..." << std::endl;
    svr.listen("0.0.0.0", port);
    return 0;
}
