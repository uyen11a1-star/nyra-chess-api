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
const char* output_names[] = {"policy", "value"};

// --- OPENING BOOK ---
std::unordered_map<std::string, std::string> opening_book = {
    {"r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq -", "a7a6"},
    {"r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -", "c2c3"},
    {"rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -", "d2d4"},
    {"rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq -", "b1c3"},
    {"rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq -", "d7d5"}
};

uint64_t zobrist_pieces[7][2][64];
uint64_t zobrist_turn;
enum TTFlag { EXACT, LOWERBOUND, UPPERBOUND };
struct TTEntry { int depth; int score; TTFlag flag; Move best_move; };
std::unordered_map<uint64_t, TTEntry> transposition_table;

std::chrono::steady_clock::time_point start_time;
bool time_is_up = false;
const int TIME_LIMIT_MS = 4000; 
const int MAX_DEPTH = 7; // Tăng Depth lên 7 vì hàm eval C++ rất nhanh

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
}

// ============================================================
// HÀM EVAL XỊN (PORT TỪ BẢN PYTHON SANG)
// ============================================================
int center_bonus(int sq) {
    int file_idx = sq % 8;
    int rank_idx = sq / 8;
    int distance = std::abs(3.5 - file_idx) + std::abs(3.5 - rank_idx);
    return static_cast<int>(14 - 4 * distance);
}

int evaluate_white_score(Board& board) {
    int score = 0;
    int phase = 0;
    
    // Đếm phase (Quân nặng còn nhiều -> Trung cuộc)
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

    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.at(sq);
        if (p == Piece::NONE) continue;

        int sign = (p.color() == Color::WHITE) ? 1 : -1;
        int pt_idx = static_cast<int>(p.type());
        int val = PIECE_VALUES[pt_idx];
        int file = sq % 8;
        int rank = sq / 8;
        int relative_rank = (p.color() == Color::WHITE) ? rank : (7 - rank);

        Square s = static_cast<Square>(sq);
        Color enemy = (p.color() == Color::WHITE) ? Color::BLACK : Color::WHITE;

        // 1. Phạt quân bị treo
        bool is_attacked = board.isAttacked(s, enemy);
        bool is_defended = board.isAttacked(s, p.color());
        if (is_attacked && !is_defended && p.type() != PieceType::KING) {
            val -= static_cast<int>(PIECE_VALUES[pt_idx] * 0.8);
        }

        // 2. Logic từng quân cờ
        if (p.type() == PieceType::PAWN) {
            val += relative_rank * 5;
            if (file == 3 || file == 4) val += 5;
            
            // Passed pawn (Tốt thông)
            bool blocked = false;
            for (int e_sq = 0; e_sq < 64; e_sq++) {
                Piece ep = board.at(e_sq);
                if (ep != Piece::NONE && ep.type() == PieceType::PAWN && ep.color() != p.color()) {
                    int ef = e_sq % 8;
                    int er = e_sq / 8;
                    int e_rel_rank = (p.color() == Color::BLACK) ? er : (7 - er);
                    if (std::abs(ef - file) <= 1 && e_rel_rank > relative_rank) {
                        blocked = true; break;
                    }
                }
            }
            if (!blocked) val += 8 + relative_rank * relative_rank * 2;

        } else if (p.type() == PieceType::KNIGHT) {
            val += center_bonus(sq);
            if (relative_rank >= 2 && relative_rank <= 5 && is_defended) {
                // Knight Outpost đơn giản
                val += 25;
            }
        } else if (p.type() == PieceType::BISHOP) {
            val += center_bonus(sq) / 2;
        } else if (p.type() == PieceType::ROOK) {
            if (relative_rank == 6) val += 15; // Rook on 7th rank
        } else if (p.type() == PieceType::QUEEN) {
            if (board.fullMoveNumber() <= 10) val -= 8; // Phạt Hậu ra sớm
        }

        score += sign * val;
    }

    // 3. Cặp Tượng
    int w_bishops = 0, b_bishops = 0;
    for(int sq=0; sq<64; sq++) {
        Piece p = board.at(sq);
        if(p != Piece::NONE && p.type() == PieceType::BISHOP) {
            if(p.color()==Color::WHITE) w_bishops++; else b_bishops++;
        }
    }
    if (w_bishops >= 2) score += 35;
    if (b_bishops >= 2) score -= 35;

    // 4. An toàn Vua & Nhập thành
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
            score -= sign * 45; // Phạt vua đi lang thang
        }

        if (!endgame) {
            if ((kf == 3 || kf == 4) && board.fullMoveNumber() > 8) score -= sign * 20;
        }
    }

    return score;
}

int evaluate_board(Board& board) {
    if (board.isGameOver().second == GameResult::LOSE) return -MATE_SCORE;
    if (board.isGameOver().second == GameResult::DRAW) return 0;
    int score = evaluate_white_score(board);
    return (board.sideToMove() == Color::WHITE) ? score : -score;
}

// --- SẮP XẾP NƯỚC ĐI BẰNG NEURAL NETWORK (Move Ordering) ---
void order_moves_with_nn(Board& board, Movelist& moves) {
    std::vector<float> inputs(18 * 8 * 8);
    encode_board(board, inputs);
    std::array<int64_t, 4> shape{1, 18, 8, 8};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, inputs.data(), inputs.size(), shape.data(), shape.size());
    
    auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 2);
    float* logits = outputs[0].GetTensorMutableData<float>(); 

    struct SM { Move m; float s; };
    std::vector<SM> sm;
    sm.reserve(moves.size());
    for (int i = 0; i < moves.size(); i++) {
        int idx = moves[i].from().index() * 64 + moves[i].to().index();
        // Điểm từ mạng Neural + Điểm thưởng nếu ăn quân
        float s = logits[idx] + (board.isCapture(moves[i]) ? 1000.0f : 0.0f); 
        sm.push_back({moves[i], s});
    }
    std::sort(sm.begin(), sm.end(), [](const SM& a, const SM& b) { return a.s > b.s; });
    for (int i = 0; i < moves.size(); i++) moves[i] = sm[i].m;
}

void order_moves_mvv_lva(Board& board, Movelist& moves) {
    struct SM { Move m; int s; };
    std::vector<SM> sm;
    sm.reserve(moves.size());
    for (int i = 0; i < moves.size(); i++) {
        int score = 0;
        if (board.isCapture(moves[i])) {
            Piece victim = board.at(moves[i].to());
            if (victim != Piece::NONE) score = 10 * PIECE_VALUES[static_cast<int>(victim.type())] - PIECE_VALUES[static_cast<int>(board.at(moves[i].from()).type())];
            else score = 1000; // En passant
        }
        sm.push_back({moves[i], score});
    }
    std::sort(sm.begin(), sm.end(), [](const SM& a, const SM& b) { return a.s > b.s; });
    for (int i = 0; i < moves.size(); i++) moves[i] = sm[i].m;
}

int quiescence(Board& board, int alpha, int beta) {
    if (time_is_up) return evaluate_board(board);
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
    if (time_is_up) return evaluate_board(board);

    static int nodes = 0;
    nodes++;
    if ((nodes & 2047) == 0) { 
        auto now = std::chrono::steady_clock::now(); 
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= TIME_LIMIT_MS) {
            time_is_up = true;
            return evaluate_board(board);
        }
    }

    if (depth == 0) return quiescence(board, alpha, beta);

    bool in_check = board.inCheck();
    if (transposition_table.size() > 100000) transposition_table.clear();

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
    
    if (root) order_moves_with_nn(board, moves);
    else order_moves_mvv_lva(board, moves);
    
    int best = -INF;
    Move best_move_in_node = moves[0];
    TTFlag flag = UPPERBOUND;

    for (int i = 0; i < moves.size(); i++) {
        board.makeMove(moves[i]);
        int extension = board.inCheck() ? 1 : 0;
        int score;
        int new_depth = depth - 1 + extension;

        // LMR
        if (i >= 3 && depth >= 3 && !in_check && !board.isCapture(moves[i]) && static_cast<int>(board.givesCheck(moves[i])) == 0) {
            new_depth -= 1; 
        }

        // PVS
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
        try {
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
                        res.set_content("{\"uci\":\"" + book_move_uci + "\", \"search\":{\"depth\":0, \"method\":\"Opening Book\"}}", "application/json");
                        return;
                    }
                }
            }

            transposition_table.clear();
            transposition_table.reserve(100000); 
            order_moves_with_nn(board, moves);
            
            int best_score = -INF;
            Move best_move = moves[0];
            int completed_depth = 0;

            time_is_up = false;
            start_time = std::chrono::steady_clock::now();

            for (int depth = 1; depth <= MAX_DEPTH; depth++) {
                if (time_is_up) break;

                int current_best_score = -INF;
                Move current_best_move = moves[0];
                
                int alpha = -INF;
                int beta = INF;
                if (depth >= 4 && completed_depth > 0) {
                    alpha = std::max(-INF, best_score - 50);
                    beta = std::min(INF, best_score + 50);
                }

                while (true) {
                    bool break_while = false;
                    for (int i = 0; i < moves.size(); i++) {
                        board.makeMove(moves[i]);
                        int score = -negamax(board, depth - 1, -beta, -alpha);
                        board.unmakeMove(moves[i]);

                        if (time_is_up) { break_while = true; break; }

                        if (score > current_best_score) {
                            current_best_score = score;
                            current_best_move = moves[i];
                        }
                        if (score > alpha) alpha = score;
                        if (alpha >= beta) break;
                    }
                    
                    if (break_while || time_is_up) break;

                    if (current_best_score <= best_score - 50) {
                        beta = alpha + 1; alpha = -INF;
                    } else if (current_best_score >= best_score + 50) {
                        alpha = beta - 1; beta = INF;
                    } else break;
                }

                if (!time_is_up) {
                    best_move = current_best_move;
                    best_score = current_best_score;
                    completed_depth = depth;
                }
            }

            auto t1 = std::chrono::steady_clock::now(); 
            int ms = (int)std::chrono::duration<double, std::milli>(t1 - start_time).count();
            
            std::string uci = uci::moveToUci(best_move);
            std::string json_resp = "{\"uci\":\"" + uci + "\", \"search\":{\"depth\":" + std::to_string(completed_depth) + ", \"time_ms\":" + std::to_string(ms) + "}}";
            
            std::cout << "FEN: " << fen_short << " -> " << uci << " (Depth: " << completed_depth << ", " << ms << "ms)" << std::endl;
            res.set_content(json_resp, "application/json");
            
        } catch (const std::exception& e) {
            std::cerr << "Lỗi C++: " << e.what() << std::endl;
            res.status = 500; res.set_content("{\"error\":\"Internal Engine Error\"}", "application/json");
        }
    });

    std::cout << "Server C++ đang chạy ở cổng 8081..." << std::endl;
    svr.listen("0.0.0.0", 8081);
    delete session;
    return 0;
}
