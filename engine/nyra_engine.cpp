#include "chess.hpp"
#if !defined(NYRA_WASM) && !defined(NYRA_TUNE)
#include "httplib.h"
#include <thread>
#include <mutex>
#endif
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

// ================= THAM SỐ ĐÁNH GIÁ (bảng PST: hàng 8 nằm trên cùng, trắng dùng sq^56, đen dùng sq) =================
// ===== EVAL_PARAMS_BEGIN (các hệ số đánh giá; có thể tinh chỉnh bằng dữ liệu) =====
int PAWN_MG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
      0,  66,   0,  16,  16,   0,  66,   0,
     12,  14,  18,  52,  52,  18,  14,  12,
      1,  17,  10,  29,  29,  10,  17,   1,
     -6, -12,   6,  10,  10,   6, -12,  -6,
     -7,  -1,  -6, -14, -14,  -6,  -1,  -7,
    -19,  -2, -10, -34, -34, -10,  -2, -19,
      0,   0,   0,   0,   0,   0,   0,   0};
int PAWN_EG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    128, 112,  60,  92,  92,  60, 112, 128,
     46,  54,  46,  30,  30,  46,  54,  46,
      2,  -6, -16,   6,   6, -16,  -6,   2,
    -16, -12, -14,   8,   8, -14, -12, -16,
    -23,  -9, -17,  -1,  -1, -17,  -9, -23,
     -8,   4,  22,  52,  52,  22,   4,  -8,
      0,   0,   0,   0,   0,   0,   0,   0};
int KNIGHT_MG[64] = {
    -150,  58, -22,  36,  36, -22,  58, -150,
    -56, -120,  76,  92,  92,  76, -120, -56,
    -54,  64,  -6,  53,  53,  -6,  64, -54,
     40,  -3,  35,  24,  24,  35,  -3,  40,
      0,  32,  11,  12,  12,  11,  32,   0,
    -18,  25,   4,  25,  25,   4,  25, -18,
    -22, -20,   8,   7,   7,   8, -20, -22,
    -72, -32, -48, -14, -14, -48, -32, -72};
int KNIGHT_EG[64] = {
    -150,   2, -22, -74, -74, -22,   2, -150,
    -60, -32, -64,  -6,  -6, -64, -32, -60,
      6,  -6,  16,  43,  43,  16,  -6,   6,
    -10,  37,  13,  10,  10,  13,  37, -10,
     -6,   2,  19,  34,  34,  19,   2,  -6,
     -8, -17,  -2,  13,  13,  -2, -17,  -8,
    -76,  18, -36,   3,   3, -36,  18, -76,
    -54, -38, -74,  26,  26, -74, -38, -54};
int BISHOP_MG[64] = {
    -38, -58, -98, -24, -24, -98, -58, -38,
    -22,  -8, -52, -90, -90, -52,  -8, -22,
    -42,  32,   3, -18, -18,   3,  32, -42,
     -8,  -5,   9,  16,  16,   9,  -5,  -8,
      6, -10,   2,  16,  16,   2, -10,   6,
      2,   6,  24,   2,   2,  24,   6,   2,
      8,  27,  22,   4,   4,  22,  27,   8,
      2,   2, -16, -42, -42, -16,   2,   2};
int BISHOP_EG[64] = {
    -44, -102,  74, -18, -18,  74, -102, -44,
     -6, -20, -28,  56,  56, -28, -20,  -6,
     34,  36,  51,  -2,  -2,  51,  36,  34,
     54,  63,  -1,   4,   4,  -1,  63,  54,
     14,  -8,  -6,  18,  18,  -6,  -8,  14,
     40, -30, -22,  24,  24, -22, -30,  40,
     24,  -7,   0,   2,   2,   0,  -7,  24,
    -10,  10,   8, -24, -24,   8,  10, -10};
int ROOK_MG[64] = {
     80,  12,   4, 108, 108,   4,  12,  80,
    -35,  50,  74,  44,  44,  74,  50, -35,
    -93,  10,  30,  10,  10,  30,  10, -93,
      5,  38, -14,  46,  46, -14,  38,   5,
    -49, -40, -16,  18,  18, -16, -40, -49,
    -35,   8, -40, -62, -62, -40,   8, -35,
    -41,  -8,  -8,  38,  38,  -8,  -8, -41,
     -6,  -4,  16,  31,  31,  16,  -4,  -6};
int ROOK_EG[64] = {
      8,  38,  42,  38,  38,  42,  38,   8,
     57,  16,  16, -32, -32,  16,  16,  57,
     67,  20,   4,  16,  16,   4,  20,  67,
     15,  22,  24,   8,   8,  24,  22,  15,
     35,  52,  40, -12, -12,  40,  52,  35,
     27,   2,  -4,  48,  48,  -4,   2,  27,
     31,  10,  26, -18, -18,  26,  10,  31,
    -12, -18,  12,  -7,  -7,  12, -18, -12};
int QUEEN_MG[64] = {
     64, -74, -88,  67,  67, -88, -74,  64,
     22, -52,   4,  20,  20,   4, -52,  22,
     12,  30, -21, -15, -15, -21,  30,  12,
     27, -42,   5, -19, -19,   5, -42,  27,
     14, -18, -11, -25, -25, -11, -18,  14,
     -6,   1,   5,   5,   5,   5,   1,  -6,
      2,  16,  25,  22,  22,  25,  16,   2,
    -42,  22,   2,  11,  11,   2,  22, -42};
int QUEEN_EG[64] = {
     46,  34, -124,   9,   9, -124,  34,  46,
      6,  66,  10, 120, 120,  10,  66,   6,
     34,  32,  59, 149, 149,  59,  32,  34,
     35, 138, 117,  11,  11, 117, 138,  35,
     38,  18,  49,  59,  59,  49,  18,  38,
     28,  13,  35,  27,  27,  35,  13,  28,
     28, -92, -69,  -2,  -2, -69, -92,  28,
    -30,  28, -10, -17, -17, -10,  28, -30};
int KING_MG[64] = {
    -44, -40,  -2, -58, -58,  -2, -40, -44,
    -72, -14, -16,   2,   2, -16, -14, -72,
    -132, -50,  36, -34, -34,  36, -50, -132,
    -88,  20,  32,  28,  28,  32,  20, -88,
    -94, -46, -30, -62, -62, -30, -46, -94,
    -20, -44, -42, -86, -86, -42, -44, -20,
     42,  22, -20, -28, -28, -20,  22,  42,
     52,  44,   0,  12,  12,   0,  44,  52};
int KING_EG[64] = {
      2,  10,  28, -84, -84,  28,  10,   2,
     62, 124,  98,  60,  60,  98, 124,  62,
     76,  94,  44,  84,  84,  44,  94,  76,
     32,  84,  52,   0,   0,  52,  84,  32,
    -72,  -4,  16,  30,  30,  16,  -4, -72,
    -46,  -6,   6,  18,  18,   6,  -6, -46,
    -26, -34, -28,  -2,  -2, -28, -34, -26,
    -38, -40, -28, -50, -50, -28, -40, -38};
int* PST_MG[6] = {PAWN_MG, KNIGHT_MG, BISHOP_MG, ROOK_MG, QUEEN_MG, KING_MG};
int* PST_EG[6] = {PAWN_EG, KNIGHT_EG, BISHOP_EG, ROOK_EG, QUEEN_EG, KING_EG};
int EV_MG[6] = {100, 380, 375, 485, 1095, 0};
int EV_EG[6] = {70, 330, 360, 590, 1085, 0};
int PASSED_MG[8] = {0, 0, 1, 14, 2, 30, 8, 0};
int PASSED_EG[8] = {0, 2, 1, 43, 81, 107, 202, 0};
int MOB_MG[4] = {6, 7, 4, 2};
int MOB_EG[4] = {1, 4, 1, 11};
int P_ISO_MG = 10;
int P_ISO_EG = 9;
int P_PROT_MG = 2;
int P_PROT_EG = 16;
int P_PASSKING = 10;
int P_OUT_MG = 36;
int P_OUT_EG = 2;
int P_ROOK_OPEN_MG = 43;
int P_ROOK_OPEN_EG = 7;
int P_ROOK_SEMI_MG = 24;
int P_ROOK_SEMI_EG = 16;
int P_ROOK_7_MG = 5;
int P_ROOK_7_EG = 23;
int P_BPAIR_MG = 60;
int P_BPAIR_EG = 65;
int P_KSH_W = 8;
int P_KSH_B = 8;
int P_KOPEN_SEMI = 22;
int P_KOPEN_OPEN = 42;
int P_KDIV = 2;
int P_TEMPO = 40;
// ===== EVAL_PARAMS_END =====
const PieceType PTS[6] = {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                          PieceType::ROOK, PieceType::QUEEN, PieceType::KING};
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
                mg[c] += EV_MG[pt] + PST_MG[pt][idx];
                eg[c] += EV_EG[pt] + PST_EG[pt][idx];
                int rel_rank = (c == 0) ? sq / 8 : 7 - sq / 8;
                int file = sq % 8;

                if (pt == 0) {  // TỐT
                    if (!(PASSED_MASK[c][sq] & pawns[e])) {
                        mg[c] += PASSED_MG[rel_rank];
                        eg[c] += PASSED_EG[rel_rank];
                        int stop = (c == 0) ? sq + 8 : sq - 8;
                        if (stop >= 0 && stop < 64) {
                            eg[c] += (cheb(ksq[e], stop) - cheb(ksq[c], stop)) * rel_rank * P_PASSKING / 2;
                        }
                    }
                    if (!(ADJ_FILES[file] & pawns[c])) { mg[c] -= P_ISO_MG; eg[c] -= P_ISO_EG; }  // tốt cô lập
                    if ((pawn_att[c] >> sq) & 1) { mg[c] += P_PROT_MG; eg[c] += P_PROT_EG; }          // tốt được tốt bảo vệ
                    continue;
                }
                if (pt == 5) continue;  // VUA xử lý riêng

                U64 att = 0;
                if (pt == 1) {
                    npm[c] += VAL[1]; phase += 1;
                    att = attacks::knight(Square(sq)).getBits();
                    int mob = popcnt(att & ~own & ~pawn_att[e]);
                    mg[c] += (mob - 4) * MOB_MG[0]; eg[c] += (mob - 4) * MOB_EG[0];
                    if (rel_rank >= 3 && rel_rank <= 5 && !(ADJ_FRONT[c][sq] & pawns[e]) &&
                        ((pawn_att[c] >> sq) & 1)) { mg[c] += P_OUT_MG; eg[c] += P_OUT_EG; }  // mã tiền đồn
                } else if (pt == 2) {
                    npm[c] += VAL[2]; phase += 1; bishops[c]++;
                    att = attacks::bishop(Square(sq), occ).getBits();
                    int mob = popcnt(att & ~own & ~pawn_att[e]);
                    mg[c] += (mob - 6) * MOB_MG[1]; eg[c] += (mob - 6) * MOB_EG[1];
                } else if (pt == 3) {
                    npm[c] += VAL[3]; phase += 2;
                    att = attacks::rook(Square(sq), occ).getBits();
                    int mob = popcnt(att & ~own);
                    mg[c] += (mob - 7) * MOB_MG[2]; eg[c] += (mob - 7) * MOB_EG[2];
                    if (!(pawns[c] & FILE_MASK[file])) {
                        if (!(pawns[e] & FILE_MASK[file])) { mg[c] += P_ROOK_OPEN_MG; eg[c] += P_ROOK_OPEN_EG; }
                        else { mg[c] += P_ROOK_SEMI_MG; eg[c] += P_ROOK_SEMI_EG; }
                    }
                    if (rel_rank == 6) { mg[c] += P_ROOK_7_MG; eg[c] += P_ROOK_7_EG; }
                } else {
                    npm[c] += VAL[4]; phase += 4;
                    att = attacks::queen(Square(sq), occ).getBits();
                    int mob = popcnt(att & ~own);
                    mg[c] += (mob - 14) * MOB_MG[3]; eg[c] += (mob - 14) * MOB_EG[3];
                }
                if (att & kz[e]) { atk_cnt[c]++; atk_units[c] += ATK_W[pt]; }
            }
        }

        if (bishops[c] >= 2) { mg[c] += P_BPAIR_MG; eg[c] += P_BPAIR_EG; }

        // Vua: tốt che chắn, cột mở quanh vua
        int kf = ksq[c] % 8, kr = ksq[c] / 8;
        int rel_k = (c == 0) ? kr : 7 - kr;
        U64 tri = (FILE_MASK[kf] | ADJ_FILES[kf]) & RANK_MASK[kr];
        U64 shield_sq = (c == 0) ? ((tri << 8) | (tri << 16)) : ((tri >> 8) | (tri >> 16));
        int shield = popcnt(shield_sq & pawns[c]);
        if (rel_k <= 1) mg[c] += std::min(shield, 4) * P_KSH_W + P_KSH_B;
        for (int ff = std::max(0, kf - 1); ff <= std::min(7, kf + 1); ff++) {
            if (!(pawns[c] & FILE_MASK[ff])) mg[c] -= (pawns[e] & FILE_MASK[ff]) ? P_KOPEN_SEMI : P_KOPEN_OPEN;
        }
    }
    // nguy hiểm khi bị nhiều quân tấn công vùng vua
    for (int c = 0; c < 2; c++) {
        int e = c ^ 1;
        if (atk_cnt[e] >= 2) mg[c] -= std::min(atk_units[e] * atk_units[e] / P_KDIV, 300);
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
    return stm_score + P_TEMPO;
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
Move countermoves[64][64];
Move prev_move[MAX_PLY + 3];
int lmr_table[64][64];
Move root_best = Move::NO_MOVE;
Move root_pv_move = Move::NO_MOVE;
Move excluded_move = Move::NO_MOVE;  // dùng khi phân tích: loại một nước ở gốc để tìm nước tốt thứ hai
bool force_full_search = false;       // dùng khi phân tích: vẫn tìm kể cả khi chỉ có 1 nước hợp lệ
#if !defined(NYRA_WASM) && !defined(NYRA_TUNE)
std::mutex engine_mutex;
#endif

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

// ================= SEE (static exchange evaluation) =================
// true nếu chuỗi đổi quân trên ô đích của nước m có lợi >= threshold
bool see_ge(const Board& b, const Move& m, int threshold) {
    if (m.typeOf() != Move::NORMAL) return threshold <= 0;
    const int from = m.from().index(), to = m.to().index();
    int swap = VAL[(int)b.at(m.to()).type()] - threshold;
    if (b.at(m.to()) == Piece::NONE) swap = -threshold;
    if (swap < 0) return false;
    swap = VAL[(int)b.at(m.from()).type()] - swap;
    if (swap <= 0) return true;
    U64 occ = b.occ().getBits() ^ (1ULL << from) ^ (1ULL << to);
    Color stm = b.sideToMove();
    const U64 bq = (b.pieces(PieceType::BISHOP) | b.pieces(PieceType::QUEEN)).getBits();
    const U64 rq = (b.pieces(PieceType::ROOK) | b.pieces(PieceType::QUEEN)).getBits();
    U64 att = (attacks::attackers(b, Color::WHITE, m.to()) | attacks::attackers(b, Color::BLACK, m.to())).getBits();
    int res = 1;
    while (true) {
        stm = ~stm;
        att &= occ;
        U64 sa = att & b.us(stm).getBits();
        if (!sa) break;
        res ^= 1;
        U64 bb;
        if ((bb = sa & b.pieces(PieceType::PAWN).getBits())) {
            if ((swap = VAL[0] - swap) < res) break;
            occ ^= bb & -bb;
            att |= attacks::bishop(m.to(), Bitboard(occ)).getBits() & bq;
        } else if ((bb = sa & b.pieces(PieceType::KNIGHT).getBits())) {
            if ((swap = VAL[1] - swap) < res) break;
            occ ^= bb & -bb;
        } else if ((bb = sa & b.pieces(PieceType::BISHOP).getBits())) {
            if ((swap = VAL[2] - swap) < res) break;
            occ ^= bb & -bb;
            att |= attacks::bishop(m.to(), Bitboard(occ)).getBits() & bq;
        } else if ((bb = sa & b.pieces(PieceType::ROOK).getBits())) {
            if ((swap = VAL[3] - swap) < res) break;
            occ ^= bb & -bb;
            att |= attacks::rook(m.to(), Bitboard(occ)).getBits() & rq;
        } else if ((bb = sa & b.pieces(PieceType::QUEEN).getBits())) {
            if ((swap = VAL[4] - swap) < res) break;
            occ ^= bb & -bb;
            att |= (attacks::bishop(m.to(), Bitboard(occ)).getBits() & bq) | (attacks::rook(m.to(), Bitboard(occ)).getBits() & rq);
        } else {
            return (att & ~b.us(stm).getBits()) ? (res ^ 1) : res;
        }
    }
    return res != 0;
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
            if (!promo && !see_ge(board, m, 0)) continue;  // bắt lỗ vốn (SEE)
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

    if (!in_check && tt_move == Move::NO_MOVE && depth >= 4 && ply > 0) depth--;  // IIR
    int static_eval = in_check ? -INF : evaluate(board);

    // --- Cắt tỉa trước khi duyệt nước đi ---
    if (!pv && !in_check && ply > 0) {
        if (depth <= 6 && static_eval - 80 * depth >= beta && std::abs(beta) < MATE_BOUND)
            return static_eval;  // reverse futility
        if (allow_null && depth >= 3 && static_eval >= beta && has_non_pawn(board, stm_c)) {
            int R = 3 + depth / 4 + std::min(3, std::max(0, (static_eval - beta) / 200));
            prev_move[ply + 1] = Move::NO_MOVE;
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
            bool bad = !see_ge(board, m, 0);
            s = (bad ? -200000 : 500000) + 10 * VAL[v] - VAL[a];
        } else if (m.typeOf() == Move::PROMOTION) {
            s = 600000 + (m.promotionType() == PieceType::QUEEN ? 1000 : 0);
        } else if (m == killers[ply][0]) s = 90000;
        else if (m == killers[ply][1]) s = 80000;
        else if (prev_move[ply] != Move::NO_MOVE && m == countermoves[prev_move[ply].from().index()][prev_move[ply].to().index()]) s = 70000;
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
        if (ply == 0 && m == excluded_move) continue;

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

        if (!pv && !in_check && ply > 0 && depth <= 6 && best > -MATE_BOUND && move_count > 1 && !gives_check && !promo) {
            if (!see_ge(board, m, capture ? -90 * depth : -50 * depth)) continue;  // SEE pruning
        }
        prev_move[ply + 1] = m;
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
                if (prev_move[ply] != Move::NO_MOVE) countermoves[prev_move[ply].from().index()][prev_move[ply].to().index()] = m;
                for (int k = 0; k < nq; k++) update_history(stm, quiets[k], -bonus);
            }
            break;
        }
        if (quiet && nq < 64) quiets[nq++] = m;
    }

    if (!stop_search && !(ply == 0 && excluded_move != Move::NO_MOVE)) {
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
    for (int i = 0; i < 64; i++) for (int j = 0; j < 64; j++) countermoves[i][j] = Move::NO_MOVE;
    for (int i = 0; i < MAX_PLY + 3; i++) prev_move[i] = Move::NO_MOVE;
    root_pv_move = Move::NO_MOVE;

    Movelist root_moves;
    movegen::legalmoves(root_moves, board);
    Move best = root_moves[0];
    int best_score = 0, done_depth = 0;
    if (root_moves.size() == 1 && !force_full_search) return {best, 0, 0};

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

#if !defined(NYRA_WASM) && !defined(NYRA_TUNE)
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
    for (int i = 0; i < 64; i++) for (int j = 0; j < 64; j++) countermoves[i][j] = Move::NO_MOVE;
    for (int i = 0; i < MAX_PLY + 3; i++) prev_move[i] = Move::NO_MOVE;
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

#endif  // PONDERING

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

#if !defined(NYRA_WASM) && !defined(NYRA_TUNE)
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

#elif defined(NYRA_WASM)  // ================= BẢN WEBASSEMBLY (chạy trong trình duyệt, không cần server) =================
static std::string g_result;

extern "C" {
__attribute__((export_name("nyra_init"))) void nyra_init() {
    init_masks();
    init_search_tables();
    init_book();
    tt.assign((size_t)1 << TT_BITS, TTEntry());
}
__attribute__((export_name("nyra_alloc"))) char* nyra_alloc(int n) { return (char*)std::malloc(n); }
__attribute__((export_name("nyra_free"))) void nyra_free(char* p) { std::free(p); }

// Nhận FEN + thời gian nghĩ (ms), trả về chuỗi JSON giống hệt API /move
// Đường đi chính lấy từ bảng băm (kiểm tra từng nước có hợp lệ)
static std::string pv_from_tt(Board b, int maxlen) {
    std::string out;
    for (int i = 0; i < maxlen; i++) {
        const TTEntry& e = tt[b.hash() & TT_MASK];
        if (e.key != b.hash() || e.move == Move::NO_MOVE) break;
        Movelist ml;
        movegen::legalmoves(ml, b);
        bool ok = false;
        for (int k = 0; k < ml.size(); k++) if (ml[k] == e.move) { ok = true; break; }
        if (!ok) break;
        if (!out.empty()) out += ",";
        out += "\"" + uci::moveToUci(e.move) + "\"";
        b.makeMove(e.move);
    }
    return out;
}

// Phân tích một thế cờ: nước tốt nhất, điểm (theo phía đang đi), điểm nước tốt thứ hai, đường đi chính, nước trong sách
__attribute__((export_name("nyra_analyze"))) const char* nyra_analyze(const char* fen_c, int time_ms) {
    std::string fen = fen_c;
    Board board = Board(fen);
    Movelist moves;
    movegen::legalmoves(moves, board);
    if (moves.size() == 0) { g_result = std::string("{\"over\":true,\"mated\":") + (board.inCheck() ? "true" : "false") + "}"; return g_result.c_str(); }

    int t = std::max(50, time_ms);
    TIME_LIMIT_MS = t; SOFT_LIMIT_MS = t;
    force_full_search = true;
    excluded_move = Move::NO_MOVE;
    SearchResult r = search_position(board);
    std::string pv = pv_from_tt(board, 8);
    bool has_second = false; int second = 0;
    if (moves.size() >= 2) {
        excluded_move = r.move;
        TIME_LIMIT_MS = std::max(50, t * 6 / 10); SOFT_LIMIT_MS = TIME_LIMIT_MS;
        SearchResult r2 = search_position(board);
        excluded_move = Move::NO_MOVE;
        second = r2.score; has_second = true;
    }
    force_full_search = false;

    std::string j = "{\"best\":\"" + uci::moveToUci(r.move) + "\",\"score\":" + std::to_string(r.score) +
                    ",\"depth\":" + std::to_string(r.depth) + ",\"pv\":[" + pv + "],\"legal\":" + std::to_string(moves.size());
    if (has_second) j += ",\"second\":" + std::to_string(second);
    auto it = opening_book.find(book_key(fen));
    if (it != opening_book.end()) {
        std::string bk;
        for (const std::string& bm : it->second)
            for (int i = 0; i < moves.size(); i++)
                if (uci::moveToUci(moves[i]) == bm) { if (!bk.empty()) bk += ","; bk += "\"" + bm + "\""; break; }
        if (!bk.empty()) j += ",\"book\":[" + bk + "]";
    }
    j += "}";
    g_result = j;
    return g_result.c_str();
}

__attribute__((export_name("nyra_move"))) const char* nyra_move(const char* fen_c, int time_ms) {
    TIME_LIMIT_MS = std::max(50, time_ms);
    SOFT_LIMIT_MS = TIME_LIMIT_MS * 13 / 20;
    std::string fen = fen_c;
    Board board = Board(fen);
    Movelist moves;
    movegen::legalmoves(moves, board);
    if (moves.size() == 0) { g_result = "{\"uci\":\"NONE\"}"; return g_result.c_str(); }

    auto it = opening_book.find(book_key(fen));
    if (it != opening_book.end()) {
        std::vector<std::string> legal_book;
        for (const std::string& bm : it->second)
            for (int i = 0; i < moves.size(); i++)
                if (uci::moveToUci(moves[i]) == bm) { legal_book.push_back(bm); break; }
        if (!legal_book.empty()) {
            std::string pick = legal_book[book_rng() % legal_book.size()];
            g_result = "{\"uci\":\"" + pick + "\", \"search\":{\"depth\":0, \"method\":\"Opening Book\"}}";
            return g_result.c_str();
        }
    }
    SearchResult r = search_position(board);
    g_result = "{\"uci\":\"" + uci::moveToUci(r.move) + "\", \"search\":{\"depth\":" + std::to_string(r.depth) +
               ", \"time_ms\":" + std::to_string((int)elapsed_ms()) + ", \"nodes\":" + std::to_string(total_nodes) +
               ", \"score\":" + std::to_string(r.score) + "}}";
    return g_result.c_str();
}
}
#endif  // NYRA_WASM
