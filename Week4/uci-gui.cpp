#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cstdlib>

#include "chess.hpp"
#include "chess-engine-v1.h"

using namespace chess;
using namespace std;

int main() {
    ChessEngine chessEngine;
    
    string line;

    // The Infinite Listening Loop
    while (getline(cin, line)) {
        
        // Skip empty lines
        if (line.empty()) continue;

        istringstream iss(line);
        string command;
        iss >> command;

        // Command Routing
        if (command == "uci") {
            cout << "id name NeuralGambit" << endl;
            cout << "id author Light" << endl;
            cout << "uciok" << endl;
        } 
        else if (command == "isready") {
            cout << "readyok" << endl;
        } 
        else if (command == "ucinewgame") {
            chessEngine.loadFen(constants::STARTPOS);
        } 
        else if (command == "position") {
            // "position startpos moves e2e4 e7e5"
            string token;
            iss >> token;

            // Handle "position startpos"
            if (token == "startpos") {
                chessEngine.loadFen(constants::STARTPOS);
                iss >> token;
            } 
            // Handle "position fen <fen_string>"
            else if (token == "fen") {
                string fen = "";
                // Extract the actual FEN string
                while (iss >> token && token != "moves") {
                    fen += token + " ";
                }
                chessEngine.loadFen(fen);
            }
            
            // Parse moves if they exist
            while (iss >> token) {
                Move m = uci::uciToMove(chessEngine.board, token);
                if (m != Move::NO_MOVE) {
                    chessEngine.board.makeMove(m);
                } else {
                    cerr << "Error: Invalid move format: " << token << endl;
                }
            }
        } 
        else if (command == "go") {
            int wtime = 0, btime = 0, winc = 0, binc = 0, movetime = 0;
            string arg;
            while (iss >> arg) {
                if (arg == "wtime") iss >> wtime;
                else if (arg == "btime") iss >> btime;
                else if (arg == "winc") iss >> winc;
                else if (arg == "binc") iss >> binc;
                else if (arg == "movetime") iss >> movetime;
            }

            int timeLimit = 1000;
            if (movetime > 0) {
                timeLimit = movetime - 50; // Reserve 50ms for GUI
            } else {
                Color ourColor = chessEngine.board.sideToMove();
                int ourTime = (ourColor == Color::WHITE) ? wtime : btime;
                int ourInc = (ourColor == Color::WHITE) ? winc : binc;

                if (ourTime > 0) {
                    // allocate 5% of remaining time + 50% of the increment
                    timeLimit = ourTime / 20 + ourInc / 2;
                    if (timeLimit < 50) timeLimit = 50;
                }
            }

            Movelist moves;
            movegen::legalmoves(moves, chessEngine.board);
            if (moves.size() > 0) {
                Move bestMove = chessEngine.findBestMove(timeLimit);
                cout << "bestmove " << uci::moveToUci(bestMove) << endl;
            } else {
                cout << "bestmove (none)" << endl;
            }
        } 
        else if (command == "quit") {
            break;
        }
    }

    return 0;
}