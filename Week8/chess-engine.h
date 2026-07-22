#ifndef CHESS_ENGINE_H
#define CHESS_ENGINE_H

#define INCREMENTAL_NNUE

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <vector>
#include <array>
#include <chrono>
#include <immintrin.h>

#include "chess.hpp" // The Disservin library

using namespace chess;
using namespace std;
using namespace chrono;

const int infinity = 1000000;

class NNUE {
public:
    vector<float> input_weights; // 40961 x 256
    vector<float> hidden_weights; // 64 x 512
    vector<float> hidden_biases; // 64
    vector<float> output_weights; // 1 x 64
    vector<float> output_bias; // 1

    bool load(const string& filename) {
        ifstream file(filename, ios::binary);
        if (!file.is_open()) {
            cerr << "Could not open NNUE file " << filename << endl;
            return false;
        }
        
        auto read_layer = [&file](vector<float>& vec, size_t size) {
            vec.resize(size);
            file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
        };

        read_layer(input_weights, 40961 * 256);
        read_layer(hidden_weights, 64 * 512);
        read_layer(hidden_biases, 64);
        read_layer(output_weights, 1 * 64);
        read_layer(output_bias, 1);
        
        file.close();
        cout << "NNUE loaded successfully from " << filename << endl;
        return true;
    }
};

enum TTEntryType : uint8_t {
    EXACT = 1,  // value is exact
    BETA = 2,   // value is a lower bound (beta cutoff)
    ALPHA = 3   // value is an upper bound (alpha cutoff)
};

struct TTEntry {
    uint64_t hash;
    int score;
    int depth;
    TTEntryType type;
    Move bestMove;
};
class ChessEngine {
private:
    time_point<steady_clock> endTime;
    bool time_out;
    int nodesEvaluated;
    const int TT_SIZE = 1048576; // 1 Megabyte of entries
    vector<TTEntry> tt;
    NNUE nnue;

    struct Accumulator {
        alignas(32) std::array<float, 256> white;
        alignas(32) std::array<float, 256> black;
    };

    Accumulator accum_stack[256];
    int accum_ply = 0;

    int score_to_tt(int score, int ply) {
        if (score >= 90000) return score + ply;
        if (score <= -90000) return score - ply;
        return score;
    }

    int score_from_tt(int score, int ply) {
        if (score >= 90000) return score - ply;
        if (score <= -90000) return score + ply;
        return score;
    }

    void add_feature_white(int p_type_val, Color color, int sq, int king_sq) {
        int white_p_type = (color == Color::WHITE) ? p_type_val : (p_type_val + 5);
        int white_idx = sq + white_p_type * 64 + king_sq * 640;
        const float* white_weights = &nnue.input_weights[white_idx * 256];
        for (int i = 0; i < 256; i += 8) {
            __m256 acc = _mm256_load_ps(&accum_stack[accum_ply].white[i]);
            __m256 w = _mm256_loadu_ps(&white_weights[i]);
            _mm256_store_ps(&accum_stack[accum_ply].white[i], _mm256_add_ps(acc, w));
        }
    }

    void sub_feature_white(int p_type_val, Color color, int sq, int king_sq) {
        int white_p_type = (color == Color::WHITE) ? p_type_val : (p_type_val + 5);
        int white_idx = sq + white_p_type * 64 + king_sq * 640;
        const float* white_weights = &nnue.input_weights[white_idx * 256];
        for (int i = 0; i < 256; i += 8) {
            __m256 acc = _mm256_load_ps(&accum_stack[accum_ply].white[i]);
            __m256 w = _mm256_loadu_ps(&white_weights[i]);
            _mm256_store_ps(&accum_stack[accum_ply].white[i], _mm256_sub_ps(acc, w));
        }
    }

    void add_feature_black(int p_type_val, Color color, int sq, int king_sq_flipped) {
        int black_p_type = (color == Color::BLACK) ? p_type_val : (p_type_val + 5);
        int sq_flipped = sq ^ 56;
        int black_idx = sq_flipped + black_p_type * 64 + king_sq_flipped * 640;
        const float* black_weights = &nnue.input_weights[black_idx * 256];
        for (int i = 0; i < 256; i += 8) {
            __m256 acc = _mm256_load_ps(&accum_stack[accum_ply].black[i]);
            __m256 w = _mm256_loadu_ps(&black_weights[i]);
            _mm256_store_ps(&accum_stack[accum_ply].black[i], _mm256_add_ps(acc, w));
        }
    }

    void sub_feature_black(int p_type_val, Color color, int sq, int king_sq_flipped) {
        int black_p_type = (color == Color::BLACK) ? p_type_val : (p_type_val + 5);
        int sq_flipped = sq ^ 56;
        int black_idx = sq_flipped + black_p_type * 64 + king_sq_flipped * 640;
        const float* black_weights = &nnue.input_weights[black_idx * 256];
        for (int i = 0; i < 256; i += 8) {
            __m256 acc = _mm256_load_ps(&accum_stack[accum_ply].black[i]);
            __m256 w = _mm256_loadu_ps(&black_weights[i]);
            _mm256_store_ps(&accum_stack[accum_ply].black[i], _mm256_sub_ps(acc, w));
        }
    }

    void recompute_white_accumulator() {
        accum_stack[accum_ply].white.fill(0.0f);
        int white_king_sq = board.kingSq(Color::WHITE).index();
        for (int p_type_val = 0; p_type_val < 5; p_type_val++) {
            PieceType pt = static_cast<PieceType::underlying>(p_type_val);
            Bitboard white_pieces = board.pieces(pt, Color::WHITE);
            while (white_pieces) {
                int sq = white_pieces.pop();
                int white_idx = sq + p_type_val * 64 + white_king_sq * 640;
                const float* white_weights = &nnue.input_weights[white_idx * 256];
                for (int i = 0; i < 256; i += 8) {
                    __m256 acc = _mm256_load_ps(&accum_stack[accum_ply].white[i]);
                    __m256 w = _mm256_loadu_ps(&white_weights[i]);
                    _mm256_store_ps(&accum_stack[accum_ply].white[i], _mm256_add_ps(acc, w));
                }
            }
            Bitboard black_pieces = board.pieces(pt, Color::BLACK);
            while (black_pieces) {
                int sq = black_pieces.pop();
                int white_idx = sq + (p_type_val + 5) * 64 + white_king_sq * 640;
                const float* white_weights = &nnue.input_weights[white_idx * 256];
                for (int i = 0; i < 256; i += 8) {
                    __m256 acc = _mm256_load_ps(&accum_stack[accum_ply].white[i]);
                    __m256 w = _mm256_loadu_ps(&white_weights[i]);
                    _mm256_store_ps(&accum_stack[accum_ply].white[i], _mm256_add_ps(acc, w));
                }
            }
        }
    }

    void recompute_black_accumulator() {
        accum_stack[accum_ply].black.fill(0.0f);
        int black_king_sq = board.kingSq(Color::BLACK).index();
        int black_king_sq_flipped = black_king_sq ^ 56;
        for (int p_type_val = 0; p_type_val < 5; p_type_val++) {
            PieceType pt = static_cast<PieceType::underlying>(p_type_val);
            Bitboard white_pieces = board.pieces(pt, Color::WHITE);
            while (white_pieces) {
                int sq = white_pieces.pop();
                int sq_flipped = sq ^ 56;
                int black_idx = sq_flipped + (p_type_val + 5) * 64 + black_king_sq_flipped * 640;
                const float* black_weights = &nnue.input_weights[black_idx * 256];
                for (int i = 0; i < 256; i += 8) {
                    __m256 acc = _mm256_load_ps(&accum_stack[accum_ply].black[i]);
                    __m256 w = _mm256_loadu_ps(&black_weights[i]);
                    _mm256_store_ps(&accum_stack[accum_ply].black[i], _mm256_add_ps(acc, w));
                }
            }
            Bitboard black_pieces = board.pieces(pt, Color::BLACK);
            while (black_pieces) {
                int sq = black_pieces.pop();
                int sq_flipped = sq ^ 56;
                int black_idx = sq_flipped + p_type_val * 64 + black_king_sq_flipped * 640;
                const float* black_weights = &nnue.input_weights[black_idx * 256];
                for (int i = 0; i < 256; i += 8) {
                    __m256 acc = _mm256_load_ps(&accum_stack[accum_ply].black[i]);
                    __m256 w = _mm256_loadu_ps(&black_weights[i]);
                    _mm256_store_ps(&accum_stack[accum_ply].black[i], _mm256_add_ps(acc, w));
                }
            }
        }
    }

    void init_accumulator() {
        accum_ply = 0;
        recompute_white_accumulator();
        recompute_black_accumulator();
    }

    void update_accumulator_make(Move move, Piece moved_piece, Piece captured_piece) {
        accum_ply++;
        accum_stack[accum_ply] = accum_stack[accum_ply - 1];

        int white_king_sq = board.kingSq(Color::WHITE).index();
        int black_king_sq = board.kingSq(Color::BLACK).index();
        int black_king_sq_flipped = black_king_sq ^ 56;

        Color us = moved_piece.color();
        PieceType pt = moved_piece.type();
        int p_type_val = static_cast<int>(pt);
        int from_idx = move.from().index();
        int to_idx = move.to().index();

        bool white_king_moved = (moved_piece.type() == PieceType::KING && us == Color::WHITE);
        bool black_king_moved = (moved_piece.type() == PieceType::KING && us == Color::BLACK);

        if (white_king_moved) {
            recompute_white_accumulator();
        }
        if (black_king_moved) {
            recompute_black_accumulator();
        }

        if (!white_king_moved) {
            if (p_type_val < 5) {
                sub_feature_white(p_type_val, us, from_idx, white_king_sq);
            }
            if (move.typeOf() == Move::PROMOTION) {
                int promo_type_val = static_cast<int>(move.promotionType());
                add_feature_white(promo_type_val, us, to_idx, white_king_sq);
            } else if (p_type_val < 5) {
                add_feature_white(p_type_val, us, to_idx, white_king_sq);
            }

            if (move.typeOf() == Move::ENPASSANT) {
                int ep_captured_sq = Square(move.to().file(), move.from().rank()).index();
                sub_feature_white(0, ~us, ep_captured_sq, white_king_sq);
            } else if (captured_piece != Piece::NONE) {
                int cap_type_val = static_cast<int>(captured_piece.type());
                if (cap_type_val < 5) {
                    sub_feature_white(cap_type_val, ~us, to_idx, white_king_sq);
                }
            }

            if (move.typeOf() == Move::CASTLING) {
                int rook_from, rook_to;
                if (to_idx == Square(Square::SQ_H1).index()) { rook_from = Square(Square::SQ_H1).index(); rook_to = Square(Square::SQ_F1).index(); }
                else if (to_idx == Square(Square::SQ_A1).index()) { rook_from = Square(Square::SQ_A1).index(); rook_to = Square(Square::SQ_D1).index(); }
                else if (to_idx == Square(Square::SQ_H8).index()) { rook_from = Square(Square::SQ_H8).index(); rook_to = Square(Square::SQ_F8).index(); }
                else { rook_from = Square(Square::SQ_A8).index(); rook_to = Square(Square::SQ_D8).index(); }
                sub_feature_white(3, us, rook_from, white_king_sq);
                add_feature_white(3, us, rook_to, white_king_sq);
            }
        }

        if (!black_king_moved) {
            if (p_type_val < 5) {
                sub_feature_black(p_type_val, us, from_idx, black_king_sq_flipped);
            }
            if (move.typeOf() == Move::PROMOTION) {
                int promo_type_val = static_cast<int>(move.promotionType());
                add_feature_black(promo_type_val, us, to_idx, black_king_sq_flipped);
            } else if (p_type_val < 5) {
                add_feature_black(p_type_val, us, to_idx, black_king_sq_flipped);
            }

            if (move.typeOf() == Move::ENPASSANT) {
                int ep_captured_sq = Square(move.to().file(), move.from().rank()).index();
                sub_feature_black(0, ~us, ep_captured_sq, black_king_sq_flipped);
            } else if (captured_piece != Piece::NONE) {
                int cap_type_val = static_cast<int>(captured_piece.type());
                if (cap_type_val < 5) {
                    sub_feature_black(cap_type_val, ~us, to_idx, black_king_sq_flipped);
                }
            }

            if (move.typeOf() == Move::CASTLING) {
                int rook_from, rook_to;
                if (to_idx == Square(Square::SQ_H1).index()) { rook_from = Square(Square::SQ_H1).index(); rook_to = Square(Square::SQ_F1).index(); }
                else if (to_idx == Square(Square::SQ_A1).index()) { rook_from = Square(Square::SQ_A1).index(); rook_to = Square(Square::SQ_D1).index(); }
                else if (to_idx == Square(Square::SQ_H8).index()) { rook_from = Square(Square::SQ_H8).index(); rook_to = Square(Square::SQ_F8).index(); }
                else { rook_from = Square(Square::SQ_A8).index(); rook_to = Square(Square::SQ_D8).index(); }
                sub_feature_black(3, us, rook_from, black_king_sq_flipped);
                add_feature_black(3, us, rook_to, black_king_sq_flipped);
            }
        }
    }


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

public:
    int eval() {
        if (nnue.input_weights.empty()) {
            int K = board.pieces(PieceType::KING, Color::WHITE).count();
            int k = board.pieces(PieceType::KING, Color::BLACK).count();
            int Q = board.pieces(PieceType::QUEEN, Color::WHITE).count();
            int q = board.pieces(PieceType::QUEEN, Color::BLACK).count();
            int R = board.pieces(PieceType::ROOK, Color::WHITE).count();
            int r = board.pieces(PieceType::ROOK, Color::BLACK).count();
            int B = board.pieces(PieceType::BISHOP, Color::WHITE).count();
            int b = board.pieces(PieceType::BISHOP, Color::BLACK).count();
            int N = board.pieces(PieceType::KNIGHT, Color::WHITE).count();
            int n = board.pieces(PieceType::KNIGHT, Color::BLACK).count();
            int P = board.pieces(PieceType::PAWN, Color::WHITE).count();
            int p = board.pieces(PieceType::PAWN, Color::BLACK).count();

            return 100000 * (K - k) + 900 * (Q - q) + 500 * (R - r) + 
                   325 * (B - b) + 300 * (N - n) + 100 * (P - p);
        }
        return evalNNUE();
    }
private:

    int evalNNUE() {
        const auto& white_accumulator = accum_stack[accum_ply].white;
        const auto& black_accumulator = accum_stack[accum_ply].black;

        alignas(32) std::array<float, 512> hidden_input;
        for (int i = 0; i < 256; i++) {
            hidden_input[i] = std::max(0.0f, std::min(white_accumulator[i], 1.0f));
            hidden_input[256 + i] = std::max(0.0f, std::min(black_accumulator[i], 1.0f));
        }

        std::array<float, 64> hidden_output;
        for (int out = 0; out < 64; out++) {
            __m256 sum_vec = _mm256_setzero_ps();
            const float* weights_row = &nnue.hidden_weights[out * 512];
            
            for (int in = 0; in < 512; in += 8) {
                __m256 input_vec = _mm256_load_ps(&hidden_input[in]);
                __m256 weight_vec = _mm256_loadu_ps(&weights_row[in]);
                sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
            }
            
            alignas(32) float temp_arr[8];
            _mm256_store_ps(temp_arr, sum_vec);
            float val = nnue.hidden_biases[out] + temp_arr[0] + temp_arr[1] + temp_arr[2] + temp_arr[3] +
                        temp_arr[4] + temp_arr[5] + temp_arr[6] + temp_arr[7];
            
            hidden_output[out] = std::max(0.0f, std::min(val, 1.0f));
        }

        float final_score = nnue.output_bias[0];
        for (int i = 0; i < 64; i++) {
            final_score += hidden_output[i] * nnue.output_weights[i];
        }

        float P = std::max(0.00001f, std::min(final_score, 0.99999f));
        float score = 400.0f * std::log(P / (1.0f - P));

        return static_cast<int>(score); // White's perspective absolute score
    }

    int quiescence_search(int alpha, int beta) {
        if ((nodesEvaluated++ & 2047) == 0) checkTime();
        if (time_out) return 0;

        int curValue = eval();
        if (board.sideToMove() == Color::WHITE) {
            if (curValue >= beta) return beta;
            alpha = max(alpha, curValue);
        } else {
            if (curValue <= alpha) return alpha;
            beta = min(beta, curValue);
        }

        Movelist captures;
        movegen::legalmoves<movegen::MoveGenType::CAPTURE>(captures, board);

        int value = curValue;
        if (board.sideToMove() == Color::WHITE) {
            for (Move move : captures) {
                make_move(move);
                int eval = quiescence_search(alpha, beta);
                unmake_move(move);

                if (time_out) return 0;
                value = max(value, eval);
                alpha = max(alpha, eval);
                if (alpha >= beta) break;
            }
        } else {
            for (Move move : captures) {
                make_move(move);
                int eval = quiescence_search(alpha, beta);
                unmake_move(move);

                if (time_out) return 0;
                value = min(value, eval);
                beta = min(beta, eval);
                if (alpha >= beta) break;
            }
        }
        return value;
    }

public:
    Board board;

    ChessEngine() {
        time_out = false;
        nodesEvaluated = 0;
        tt.resize(TT_SIZE);
        nnue.load("neuralgambit.nnue");
    }

    void loadFen(string fen) {
        board = Board(fen);
        init_accumulator();
    }

    void make_move(Move move) {
        Piece moved_piece = board.at<Piece>(move.from());
        Piece captured_piece = (move.typeOf() == Move::CASTLING) ? Piece::NONE : board.at<Piece>(move.to());
        board.makeMove(move);
        update_accumulator_make(move, moved_piece, captured_piece);
    }

    void unmake_move(Move move) {
        board.unmakeMove(move);
        accum_ply--;
    }

    int getNodesEvaluated() const {
        return nodesEvaluated;
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
            make_move(move);
            int eval = -negamax_alpha_beta_pruning(depth - 1, -beta, -alpha, best_child_moves);
            unmake_move(move);
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
        int original_alpha = alpha;
        int original_beta = beta;

        uint64_t key = board.hash();
        int index = key & (TT_SIZE - 1);

        if (tt[index].hash == key && tt[index].depth >= depth) {
            int tt_score = score_from_tt(tt[index].score, accum_ply);
            if (tt[index].type == EXACT) return tt_score;
            else if (tt[index].type == BETA && tt_score >= beta) return tt_score;
            else if (tt[index].type == ALPHA && tt_score <= alpha) return tt_score;
        }
        
        if ((nodesEvaluated++ & 2047) == 0) checkTime();
        if (time_out) return 0;

        // if (board.isGameOver().second != GameResult::NONE) {
        //     return get_utility_given_terminal_state(depth);
        // }
        // Small optimization
        // board.isGameOver() generates all legal moves internally to check if any moves exist
        // Since we are looking at all legal moves in the main loop already
        // checking if the list is empty is enough to determine if the game is over
        Movelist moves;
        movegen::legalmoves(moves, board);

        if (moves.size() == 0) {
            if (board.inCheck()) {
                // Checkmate
                if (board.sideToMove() == Color::WHITE) return -100000 + accum_ply;
                else return 100000 - accum_ply;
            }
            return 0; // Stalemate
        }
        // Quick draw checks
        if (board.isHalfMoveDraw() || board.isInsufficientMaterial() || board.isRepetition()) {
            return 0;
        }

        // Quiescence search at leaf nodes to handle captures
        if (depth == 0) return quiescence_search(alpha, beta);

        // Get TT move for move ordering
        Move ttMove = Move();
        if (tt[index].hash == key) {
            ttMove = tt[index].bestMove;
        }
        if (ttMove != Move()) {
            for (int i = 0; i < moves.size(); i++) {
                if (moves[i] == ttMove) {
                    swap(moves[0], moves[i]);
                    break;
                }
            }
        }

        int value;
        Move bestMove = Move();

        if (board.sideToMove() == Color::WHITE) {
            value = -infinity;
            for (Move move : moves) {
                make_move(move);
                int eval = alpha_beta_pruning(depth - 1, alpha, beta);
                unmake_move(move);

                if (time_out) return 0;
                if (eval > value) {
                    value = eval;
                    bestMove = move;
                }
                alpha = max(alpha, eval);
                if (alpha >= beta) break;
            }
        } else {
            value = infinity;
            for (Move move : moves) {
                make_move(move);
                int eval = alpha_beta_pruning(depth - 1, alpha, beta);
                unmake_move(move);

                if (time_out) return 0;
                if (eval < value) {
                    value = eval;
                    bestMove = move;
                }
                beta = min(beta, eval);
                if (alpha >= beta) break;
            }
        }
        if (!time_out) {
            int tt_value = score_to_tt(value, accum_ply);
            if (value <= original_alpha) {
                tt[index] = {key, tt_value, depth, ALPHA, bestMove};
            }
            else if (value >= original_beta) {
                tt[index] = {key, tt_value, depth, BETA, bestMove};
            }
            else {
                tt[index] = {key, tt_value, depth, EXACT, bestMove};
            }
        }
        return value;
    }

    Move findBestMove(int timeLimit) {
        endTime = steady_clock::now() + milliseconds(timeLimit);
        time_out = false;
        nodesEvaluated = 0;
        init_accumulator();

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
                    make_move(move);
                    int eval = alpha_beta_pruning(currDepth - 1, -infinity, infinity);
                    unmake_move(move);
                    if (time_out) break;
                    if (eval > value) {
                        value = eval; currbest = move;
                    }
                }
            } else {
                int value = infinity;
                for (Move move : moves) {
                    make_move(move);
                    int eval = alpha_beta_pruning(currDepth - 1, -infinity, infinity);
                    unmake_move(move);
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
