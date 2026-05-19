# Write your code here
n = int(input())

nums = list(map(int, input().split()))

score1 = 0
score2 = 0
player = False  # 0 for Player 1, 1 for Player 2
l = 0
r = n - 1

while l <= r:
    if not player:
        if nums[l] > nums[r]:
            score1 += nums[l]
            l += 1
        else:
            score1 += nums[r]
            r -= 1
    else:
        if nums[l] > nums[r]:
            score2 += nums[l]
            l += 1
        else:
            score2 += nums[r]
            r -= 1
    
    player = not player

if score1 > score2:
    print("Player 1 wins")
elif score1 < score2:
    print("Player 2 wins")
else:
    print("Its a draw")