#include "chess.hpp"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <string>

using namespace chess;
const int PIECE_VALUES[7] = {0, 100, 320, 330, 500, 900, 20000}; 
const int MATE_SCORE = 100000;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "NyraNet");
Ort::Session* session;
const char* input_names[] = {"input"};
const char* output_names[] = {"policy"};

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

int evaluate_board(const Board& board) {
    if (board.isGameOver().second == GameResult::LOSE) return -MATE_SCORE;
    if (board.isGameOver().second == GameResult::DRAW) return 0;

    int score = 0;
    for (int sq = 0; sq < 64; sq++) {
        Piece piece = board.at(sq);
        if (piece != Piece::NONE) {
            int sign = (piece.color() == Color::WHITE) ? 1 : -1;
            int val = PIECE_VALUES[static_cast<int>(piece.type())];
            int rank = sq / 8; int file = sq % 8;
            int dist = std::abs(3.5 - rank) + std::abs(3.5 - file);
            if (piece.type() == PieceType::PAWN) {
                val += (14 - 4 * dist);
                if (piece.color() == Color::WHITE) val += rank * 5; else val += (7 - rank) * 5;
            } else if (piece.type() == PieceType::KNIGHT || piece.type() == PieceType::BISHOP) {
                if (piece.color() == Color::WHITE && rank > 0) val += 20;
                if (piece.color() == Color::BLACK && rank < 7) val += 20;
                val += (14 - 4 * dist);
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
        float s = logits[idx] + (board.isCapture(moves[i]) ? 100.0f : 0.0f);
        sm.push_back({moves[i], s});
    }
    std::sort(sm.begin(), sm.end(), [](const SM& a, const SM& b) { return a.s > b.s; });
    for (int i = 0; i < moves.size(); i++) moves[i] = sm[i].m;
}

int negamax(Board& board, int depth, int alpha, int beta, bool root = false) {
    if (depth == 0) return evaluate_board(board);
    Movelist moves; movegen::legalmoves(moves, board);
    if (moves.size() == 0) return board.inCheck() ? -MATE_SCORE + depth : 0;
    if (root) order_moves_with_nn(board, moves);
    
    int best = -MATE_SCORE * 2;
    for (int i = 0; i < moves.size(); i++) {
        board.makeMove(moves[i]);
        int score = -negamax(board, depth - 1, -beta, -alpha);
        board.unmakeMove(moves[i]);
        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    return best;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "No FEN provided" << std::endl;
            return 1;
        }
        std::string fen = argv[1];
        
        Ort::SessionOptions opts; opts.SetIntraOpNumThreads(1);
        session = new Ort::Session(env, "nyra_net.onnx", opts);

        Board board = Board(fen);
        Movelist moves; movegen::legalmoves(moves, board);
        
        if (moves.size() == 0) {
            std::cout << "NONE" << std::endl;
            delete session;
            return 0;
        }

        order_moves_with_nn(board, moves);
        int best_score = -MATE_SCORE * 2;
        Move best_move = moves[0];

        for (int i = 0; i < moves.size(); i++) {
            board.makeMove(moves[i]);
            int score = -negamax(board, 3, -MATE_SCORE * 2, MATE_SCORE * 2);
            board.unmakeMove(moves[i]);
            if (score > best_score) { best_score = score; best_move = moves[i]; }
        }

        std::cout << uci::moveToUci(best_move) << std::endl;
        delete session;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "C++ Exception: " << e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "Unknown C++ Exception" << std::endl;
        return 3;
    }
}
