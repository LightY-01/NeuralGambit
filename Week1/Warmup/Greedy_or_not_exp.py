# Brute Force: O(2^n)
n = int(input())
nums = list(map(int, input().split()))


def best_score(l, r):
    if l == r:
        return nums[l]
    
    left = nums[l] - best_score(l + 1, r)
    right = nums[r] - best_score(l, r - 1)
    
    return max(left, right)


diff = best_score(0, n - 1)

if diff > 0:
    print("Player 1 wins")
elif diff < 0:
    print("Player 2 wins")
else:
    print("Its a draw")