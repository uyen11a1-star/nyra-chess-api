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
#include <memory>

using namespace chess;

const int MATE_SCORE = 100000;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "NyraNet");
Ort::Session* session;
const char* input_names[] = {"input"};
const char* output_names[] = {"policy", "value"};

std::chrono::steady_clock::time_point start_time;
const int TIME_LIMIT_MS = 3500; 
const int MAX_SIMULATIONS = 150; // Vẫn giữ mức an toàn cho RAM

class MCTSNode {
public:
    Board board;
    MCTSNode* parent;
    Move move; 
    float prior;
    int visits;
    float value_sum;
    std::vector<std::unique_ptr<MCTSNode>> children;
    bool is_expanded;

    MCTSNode(Board b, MCTSNode* p, Move m, float pr) 
        : board(b), parent(p), move(m), prior(pr), visits(0), value_sum(0), is_expanded(false) {}

    float get_value() const {
        return visits > 0 ? value_sum / visits : 0.0f;
    }

    float get_uct(float c_puct = 1.414f) const {
        if (visits == 0) return 99999.0f;
        float q = get_value();
        float u = c_puct * prior * std::sqrt(parent->visits) / (1 + visits);
        return q + u;
    }

    MCTSNode* select_best_child() {
        float best_uct = -1e9;
        MCTSNode* best = nullptr;
        for (auto& child : children) {
            float uct = child->get_uct();
            if (uct > best_uct) {
                best_uct = uct;
                best = child.get();
            }
        }
        return best;
    }
};

void evaluate_nn(const Board& board, std::vector<float>& policy, float& value) {
    std::vector<float> inputs(18 * 8 * 8, 0.0f);
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.at(sq);
        if (p != Piece::NONE) {
            int plane = (p.color() == Color::WHITE) ? (static_cast<int>(p.type()) - 1) : (static_cast<int>(p.type()) - 1 + 6);
            inputs[plane * 64 + (sq / 8) * 8 + (sq % 8)] = 1.0f;
        }
    }
    if (board.sideToMove() == Color::WHITE) {
        for (int i = 0; i < 64; i++) inputs[12 * 64 + i] = 1.0f;
    }
    
    std::array<int64_t, 4> shape{1, 18, 8, 8};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, inputs.data(), inputs.size(), shape.data(), shape.size());
    
    auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 2);
    
    float* logits = outputs[0].GetTensorMutableData<float>();
    float* val_out = outputs[1].GetTensorMutableData<float>();
    
    policy.resize(4096);
    float max_logit = -1e9;
    for (int i = 0; i < 4096; i++) max_logit = std::max(max_logit, logits[i]);
    float sum = 0;
    for (int i = 0; i < 4096; i++) {
        policy[i] = std::exp(logits[i] - max_logit);
        sum += policy[i];
    }
    for (int i = 0; i < 4096; i++) policy[i] /= sum;
    
    value = val_out[0];
}

Move mcts_search(Board root_board) {
    std::unique_ptr<MCTSNode> root = std::make_unique<MCTSNode>(root_board, nullptr, Move(), 1.0f);
    int sims = 0;
    
    while (sims < MAX_SIMULATIONS) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= TIME_LIMIT_MS) break;
        
        MCTSNode* node = root.get();
        Board temp_board = node->board;
        
        while (node->is_expanded && !node->children.empty()) {
            node = node->select_best_child();
            temp_board = node->board;
        }
        
        std::vector<float> policy;
        float value;
        
        if (temp_board.isGameOver().second == GameResult::LOSE) {
            value = (temp_board.sideToMove() == Color::WHITE) ? -1.0f : 1.0f;
        } else if (temp_board.isGameOver().second == GameResult::DRAW) {
            value = 0.0f;
        } else {
            evaluate_nn(temp_board, policy, value);
            
            Movelist moves;
            movegen::legalmoves(moves, temp_board);
            for (int i = 0; i < moves.size(); i++) {
                Move m = moves[i];
                Board next_board = temp_board;
                next_board.makeMove(m);
                int idx = m.from().index() * 64 + m.to().index();
                node->children.push_back(std::make_unique<MCTSNode>(next_board, node, m, policy[idx]));
            }
            node->is_expanded = true;
        }
        
        while (node != nullptr) {
            node->visits++;
            node->value_sum += (node->board.sideToMove() == temp_board.sideToMove()) ? value : -value;
            node = node->parent;
        }
        sims++;
    }
    
    float best_visits = -1;
    Move best_move = root->children.empty() ? Move() : root->children[0]->move;
    for (auto& child : root->children) {
        if (child->visits > best_visits) {
            best_visits = child->visits;
            best_move = child->move;
        }
    }
    
    return best_move; 
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
    Ort::SessionOptions opts; opts.SetIntraOpNumThreads(1);
    std::cout << "Đang load mô hình ONNX (MCTS)..." << std::endl;
    session = new Ort::Session(env, "nyra_net.onnx", opts);

    httplib::Server svr;

    // THÊM ROUTE HEALTH VÀ ROOT
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content("{\"status\":\"ok\", \"engine\":\"C++ MCTS\"}", "application/json");
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content("{\"status\":\"ready\", \"model_loaded\":true}", "application/json");
    });

    svr.Options(R"(/.*)", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
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
            
            Board board = Board(fen);
            Movelist moves; movegen::legalmoves(moves, board);
            if (moves.size() == 0) {
                res.set_content("{\"uci\":\"NONE\"}", "application/json"); return;
            }

            start_time = std::chrono::steady_clock::now();
            Move best_move = mcts_search(board);
            
            if (best_move == Move()) best_move = moves[0];

            int ms = (int)std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
            
            std::string uci = uci::moveToUci(best_move);
            std::string json_resp = "{\"uci\":\"" + uci + "\", \"search\":{\"method\":\"MCTS\", \"time_ms\":" + std::to_string(ms) + "}}";
            res.set_content(json_resp, "application/json");
            
        } catch (const std::exception& e) {
            std::cerr << "Lỗi C++: " << e.what() << std::endl;
            res.status = 500; res.set_content("{\"error\":\"Internal Engine Error\"}", "application/json");
        }
    });

    std::cout << "Server C++ MCTS đang chạy ở cổng 8081..." << std::endl;
    svr.listen("0.0.0.0", 8081);
    delete session;
    return 0;
}
