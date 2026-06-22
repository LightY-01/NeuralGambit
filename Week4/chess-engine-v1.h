#ifndef CHESS_ENGINE_H
#define CHESS_ENGINE_H

#include <iostream>
#include <string>
#include <algorithm>
#include "chess.hpp" // The Disservin library
#include <chrono>

using namespace chess;
using namespace std;
using namespace chrono;

const int infinity = 1000000;

class ChessEngine {
private:
    time_point<steady_clock> endTime;
    bool time_out;
    int nodesEvaluated;

    void checkTime() {
        if (steady_clock::now() >= endTime) {
            time_out = true;
        }
    }

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

    // I will implement PST's later
    int eval() {
        // a(K - k) + b(Q - q) + c(R - r) + d(B - b) + e(N - n) + f(P - p)
        int score = 0;
        int P = board.pieces(PieceType::PAWN, Color::WHITE).count();
        int p = board.pieces(PieceType::PAWN, Color::BLACK).count();
        int N = board.pieces(PieceType::KNIGHT, Color::WHITE).count();
        int n = board.pieces(PieceType::KNIGHT, Color::BLACK).count();
        int B = board.pieces(PieceType::BISHOP, Color::WHITE).count();
        int b = board.pieces(PieceType::BISHOP, Color::BLACK).count();
        int R = board.pieces(PieceType::ROOK, Color::WHITE).count();
        int r = board.pieces(PieceType::ROOK, Color::BLACK).count();
        int Q = board.pieces(PieceType::QUEEN, Color::WHITE).count();
        int q = board.pieces(PieceType::QUEEN, Color::BLACK).count();
        int K = board.pieces(PieceType::KING, Color::WHITE).count();
        int k = board.pieces(PieceType::KING, Color::BLACK).count();

        score += 100000 * (K - k);
        score += 900 * (Q - q);
        score += 500 * (R - r);
        score += 325 * (B - b);
        score += 300 * (N - n);
        score += 100 * (P - p);
        
        return score;
    }

public:
    Board board;

    ChessEngine() {
        time_out = false;
        nodesEvaluated = 0;
    }

    void loadFen(string fen) {
        board = Board(fen);
    }

    // max(a, b) = -min(-a, -b)
    // For Mate-in-X puzzles
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
        if (nodesEvaluated++ % 2048 == 0) {
            checkTime();
        }
        if (time_out) return 0;

        Movelist moves;
        movegen::legalmoves(moves, board);   
        
        if (moves.size() == 0) {
            if (board.inCheck()) {
                // Checkmate
                if (board.sideToMove() == Color::WHITE) return -100000 - depth;
                else return 100000 + depth;
            }
            // Stalemate
            return 0;
        }

        // Quick draw checks
        if (board.isHalfMoveDraw() || board.isInsufficientMaterial() || board.isRepetition()) {
            return 0;
        }

        if (depth == 0) {
            return eval();
        }

        if (board.sideToMove() == Color::WHITE) {
            int value = -infinity;
            for (Move move : moves) {
                board.makeMove(move);
                int eval = alpha_beta_pruning(depth - 1, alpha, beta);
                board.unmakeMove(move);

                if (time_out) return 0;
                value = max(value, eval);
                alpha = max(alpha, eval);
                if (alpha >= beta) {
                    break;
                }
            }
            return value;
        } else {
            int value = infinity;
            for (Move move : moves) {
                board.makeMove(move);
                int eval = alpha_beta_pruning(depth - 1, alpha, beta);
                board.unmakeMove(move);

                if (time_out) return 0;
                value = min(value, eval);
                beta = min(beta, eval);
                if (alpha >= beta) {
                    break;
                }
            }
            return value;
        }
    }

    Move findBestMove(int timeLimit) {
        endTime = steady_clock::now() + milliseconds(timeLimit);
        time_out = false;
        nodesEvaluated = 0;

        Movelist moves;
        movegen::legalmoves(moves, board);
        if (moves.size() == 0) return Move();
        Move bestMove = moves[0];

        int currDepth = 1;
        while (true) {
            Move currbest = moves[0];
            if (board.sideToMove() == Color::WHITE) {
                int value = -infinity;

                for (Move move : moves) {
                    board.makeMove(move);
                    int eval = alpha_beta_pruning(currDepth - 1, -infinity, infinity);
                    board.unmakeMove(move);
                    if (time_out) break;
                    if (eval > value) {
                        value = eval; currbest = move;
                    }
                }
            } else {
                int value = infinity;
                for (Move move : moves) {
                    board.makeMove(move);
                    int eval = alpha_beta_pruning(currDepth - 1, -infinity, infinity);
                    board.unmakeMove(move);
                    if (time_out) break;
                    if (eval < value) {
                        value = eval; currbest = move;
                    }
                }
            }
            if (time_out) break;
            bestMove = currbest;
            currDepth++;
        }
        return bestMove;
    }
};

#endif // CHESS_ENGINE_H
