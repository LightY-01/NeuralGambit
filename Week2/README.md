This week, you will work on two tasks involving game-playing bots:

## Task 1: Tic-Tac-Toe Bot

You will create a bot to play Tic-Tac-Toe using **alpha-beta pruning**.  
- Complete the implementation in `q1.py`.  
- Test your bot using `play_tictactoe.py`.

## Task 2: Notakto Bot

Notakto is a variant of Tic-Tac-Toe played on multiple boards with only **'X'** as a symbol.  
- You will again use **alpha-beta pruning**.  
- Complete the implementation in `q2.py`.

### Observed results

For history=[], num_boards=2:
- alpha-beta pruning visited 3,894,802 histories and took about 1 minute 44 seconds.
- maxmin with memoization took about 13 seconds.
maxmin was significantly faster.

For history=[4], num_boards=2:
- alpha-beta pruning visited 25,845 histories and took about 1 second.
- maxmin with memoization took about 3 seconds.
alpha-beta pruning was faster but maxmin is also not very slow

For history=[4, 13], num_boards=2:
- alpha-beta pruning visited 104,505 histories and took about 3 seconds.
- maxmin with memoization took about 1 second.

The results show that the faster method depends on the starting history.
- Alpha-beta pruning is more effective when the move ordering is strong and the position allows early pruning. That is why positions like [4], [0], and [1] finish quickly.
- Maxmin with memoization becomes very effective when many different histories lead to the same board position. In the full game tree from [], this reuse of repeated positions helps a lot, so it can outperform alpha-beta.
- In positions where the tree is smaller or pruning is strong, alpha-beta can be faster.
- In positions with many repeated states, memoization can be faster.

So, from these experiments, there is no single method that is always faster. The better approach depends on the starting state.

## Notes

- Both template files (`q1.py` and `q2.py`) are provided.  
- Make sure to read `problem_statement.pdf` thoroughly to understand the tasks.
