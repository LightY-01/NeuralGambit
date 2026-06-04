#include "chess-engine.h"

int main() {
    ChessEngine chessEngine;
    chessEngine.loadFen("r5rk/2p1Nppp/3p3P/pp2p1P1/4P3/2qnPQK1/8/R6R w - - 1 0");
    // cout << chessEngine.board << '\n';

    // Using Negamax Alpha-Beta Pruning (Takes less time)
    vector<Move> best_moves_vec;
    int value = chessEngine.negamax_alpha_beta_pruning(8, -infinity, infinity, best_moves_vec);
    Board b = chessEngine.board;
    for (auto move : best_moves_vec) {
        cout << uci::moveToSan(b, move) << '\n';
        b.makeMove(move);
    }
    cout << b << '\n';

    // // Using Alpha-Beta Pruning (Takes more time since it is not tracking best moves vector, it recomputes it again)
    // Move bestMove;
    // for (int i = 0; i < 8; i++) {
    //     bestMove = chessEngine.findBestMove(8);
    //     // cout << uci::moveToUci(bestMove) << '\n';
    //     cout << uci::moveToSan(chessEngine.board, bestMove) << '\n';
    //     chessEngine.board.makeMove(bestMove);
    // }
    // cout << chessEngine.board << '\n';

    return 0;
}