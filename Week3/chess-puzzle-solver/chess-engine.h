#ifndef CHESS_ENGINE_H
#define CHESS_ENGINE_H

#include <iostream>
#include <string>
#include <algorithm>
#include "chess.hpp" // The Disservin library

using namespace chess;
using namespace std;

const int infinity = 1000000;

// Solves Mate-in-X puzzles
// Always White to Move
class ChessEngine {
private:
    // Utility function for terminal states
    // If Engine sees a Mate-in-1 and a Mate-in-2, adding the remaining depth to the score 
    // forces the engine to pick the fastest possible checkmate
    int get_utility_given_terminal_state(int depth) {
        // board.isGameOver() => pair<GameResultReason, GameResult>
        // GameResult => WIN, LOSE, DRAW and NONE
        if (board.isGameOver().first == GameResultReason::CHECKMATE) {
            if (board.sideToMove() == Color::WHITE) {
                return -100000 - depth; // Black wins
            }
            else {
                return 100000 + depth; // White wins
            }
        }
        // Stalemate
        return 0;
    }

    int eval() {
        return 0;
    }

public:
    Board board;

    void loadFen(string fen) {
        board = Board(fen);
    }

    // max(a, b) = -min(-a, -b)
    int negamax_alpha_beta_pruning(int depth, int alpha, int beta, vector<Move> &best_moves_vec) {
        if (board.isGameOver().second != GameResult::NONE) {
            if (board.isGameOver().first == GameResultReason::CHECKMATE) {
                return -100000 - depth;
            }
            // Stalemate
            return 0;
        }

        if (depth == 0) {
            return eval();
        }

        int value = -infinity;
        Movelist moves;
        movegen::legalmoves(moves, board);
        for (Move move : moves) {
            vector<Move> best_child_moves;
            board.makeMove(move);
            int eval = -negamax_alpha_beta_pruning(depth - 1, -beta, -alpha, best_child_moves);
            board.unmakeMove(move);
            if (eval > value) {
                value = eval;
                best_moves_vec.clear();
                best_moves_vec.push_back(move);
                best_moves_vec.insert(best_moves_vec.end(), best_child_moves.begin(), best_child_moves.end());
            }
            alpha = max(alpha, eval);
            if (alpha >= beta) {
                break;
            }
        }
        return value;
    }

    int alpha_beta_pruning(int depth, int alpha, int beta) {
        if (board.isGameOver().second != GameResult::NONE) {
            return get_utility_given_terminal_state(depth);
        }

        if (depth == 0) {
            return eval();
        }

        if (board.sideToMove() == Color::WHITE) {
            int value = -infinity;
            Movelist moves;
            movegen::legalmoves(moves, board);
            for (Move move : moves) {
                board.makeMove(move);
                int eval = alpha_beta_pruning(depth - 1, alpha, beta);
                board.unmakeMove(move);
                value = max(value, eval);
                alpha = max(alpha, eval);
                if (alpha >= beta) {
                    break;
                }
            }
            return value;
        } else {
            int value = infinity;
            Movelist moves;
            movegen::legalmoves(moves, board);
            for (Move move : moves) {
                board.makeMove(move);
                int eval = alpha_beta_pruning(depth - 1, alpha, beta);
                board.unmakeMove(move);
                value = min(value, eval);
                beta = min(beta, eval);
                if (alpha >= beta) {
                    break;
                }
            }
            return value;
        }
    }

    // Assuming provided target depth is greater than x for Mate-in-X puzzle
    Move findBestMove(int targetDepth) {
        Move bestMove;
        if (board.sideToMove() == Color::WHITE) {
            int value = -infinity;
            Movelist moves;
            movegen::legalmoves(moves, board);
            for (Move move : moves) {
                board.makeMove(move);
                int eval = alpha_beta_pruning(targetDepth - 1, -infinity, infinity);
                board.unmakeMove(move);
                if (eval > value) {
                    value = eval; bestMove = move;
                }
            }
            return bestMove;
        } else {
            int value = infinity;
            Movelist moves;
            movegen::legalmoves(moves, board);
            for (Move move : moves) {
                board.makeMove(move);
                int eval = alpha_beta_pruning(targetDepth - 1, -infinity, infinity);
                board.unmakeMove(move);
                if (eval < value) {
                    value = eval; bestMove = move;
                }
            }
            return bestMove;
        }
    }
};

#endif // CHESS_ENGINE_H
