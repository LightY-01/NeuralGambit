#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <cassert>
#include "chess-engine.h"

using namespace std;
using namespace chess;

bool run_move_test(ChessEngine& engine, const string& fen, const vector<string>& acceptable_moves, int time_limit = 500) {
    engine.loadFen(fen);
    Move best = engine.findBestMove(time_limit);
    string best_str = uci::moveToUci(best);
    
    bool found = false;
    for (const string& m : acceptable_moves) {
        if (best_str == m) {
            found = true;
            break;
        }
    }
    
    return found;
}

int main() {
    ChessEngine engine;
    
    cout << "          NeuralGambit Benchmark           " << '\n';

    // Performance Benchmark
    engine.loadFen(constants::STARTPOS);
    auto start = chrono::steady_clock::now();
    int dummy = 0;
    for (int i = 0; i < 100000; i++) {
        dummy += engine.board.sideToMove() == Color::WHITE ? engine.eval() : -engine.eval();
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    
    cout << "Time for 100,000 evaluations: " << duration << " ms" << '\n';
    cout << "NPS (evals/sec): " << (100000.0 / (duration / 1000.0)) << '\n';

    // Correctness and Tactical Tests
    cout << "            Correctness Tests              " << '\n';

    // Test 1: Scholar's Mate (Mate in 1)
    if (run_move_test(engine, "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", {"h5f7"})) {
        cout << "Scholar's Mate in 1 (PASSED)" << '\n';
    } else {
        cout << "Scholar's Mate in 1 (FAILED)" << '\n';
    }

    // Test 2: Center Pawn Capture
    if (run_move_test(engine, "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", {"e4d5"})) {
        cout << "Center Pawn Capture (PASSED)" << '\n';
    } else {
        cout << "Center Pawn Capture (FAILED)" << '\n';
    }

    // Test 3: Hanging Knight Recapture
    engine.loadFen(constants::STARTPOS);
    vector<string> game_moves = {"e2e4", "g8f6", "b1c3", "f6e4", "c3e4", "d7d5", "e4c5", "e7e6", "c5b3", "c7c5", "d2d3", "b7b5", "b3d2", "b8c6", "g1f3", "c6d4", "f3d4"};
    for (const string& m_str : game_moves) {
        Move m = uci::uciToMove(engine.board, m_str);
        engine.board.makeMove(m);
    }
    
    Move best = engine.findBestMove(500);
    string best_str = uci::moveToUci(best);
    cout << "Test [Hanging Knight Recapture]: Played " << best_str;
    if (best_str == "c5d4" || best_str == "e6d4") {
        cout << " (PASSED)" << '\n';
    } else {
        cout << " (FAILED! Left knight hanging on d4)" << '\n';
    }

    return 0;
}
