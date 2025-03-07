# 49. Group Anagrams
class Solution:
    def groupAnagrams(self, strs):
        groupedResults = {}
        for currStr in strs:
            sortedStr = ''.join(sorted(currStr))
            if sortedStr in groupedResults:
                groupedResults[sortedStr].append(currStr)
            else:
                groupedResults[sortedStr] = [currStr]
        return [groupedResults[key] for key in groupedResults]
    
# 155 Min Stack
class MinStack:

    def __init__(self):
        self.data = []

    def push(self, val: int) -> None:
        prevMin = self.getMin()
        return self.data.append([val, (prevMin if prevMin != None and prevMin < val else val)])

    def pop(self) -> None:
        return self.data.pop()[0] if self.data else None

    def top(self) -> int:
        return self.data[-1][0] if self.data else None

    def getMin(self) -> int:
        return self.data[-1][1] if self.data else None

# 134 Gas Station
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if (sum(gas) - sum(cost)) < 0:
            return -1
        currDiff = 0
        currStart = 0
        for i in range(len(gas)):
            currDiff += gas[i] - cost[i]
            if currDiff < 0:
                currStart = i+1
                currDiff = 0
        return currStart

#17 Letter Combinations of a Phone Number
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        numToLetDict = {
                            '1': [], 
                            '2': ['a','b','c'], 
                            '3': ['d','e','f'], 
                            '4': ['g','h','i'], 
                            '5': ['j','k','l'],
                            '6': ['m','n','o'],
                            '7': ['p','q','r','s'],
                            '8': ['t','u','v'],
                            '9': ['w','x','y','z']
                        }
        results = []

        for currNum in digits:
            if not results:
                results = numToLetDict[currNum]
            else:
                newResults = []
                for currChar in numToLetDict[currNum]:
                    for currEntry in results:
                        newResults.append(currEntry + currChar)
                results = newResults
        return results
    
# 35 Search Insert Position
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return bst(nums, 0, target)


def bst(subNums, leftIndex, target):
    if len(subNums) == 1:
        if subNums[0] >= target:
            return leftIndex
        else:
            return leftIndex + 1
    midIndex = len(subNums)//2
    if subNums[midIndex] == target:
        return leftIndex + midIndex
    elif subNums[midIndex] > target:
        return bst(subNums[:midIndex], leftIndex, target)
    else:
        return bst(subNums[midIndex:], leftIndex + midIndex, target)

# 48 Rotate Image
# TO DO: Slow, improve by using flip, transpose approach

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        matrixLen = len(matrix) - 1

        for j in range(ceil(len(matrix)/2)):
            for i in range(j,len(matrix)-1-j):
                temp2 = matrix[i][matrixLen - j]
                matrix[i][matrixLen - j] = matrix[j][i]
                temp1 = temp2
                temp2 = matrix[matrixLen-j][matrixLen-i]
                matrix[matrixLen-j][matrixLen-i] = temp1
                temp1 = temp2
                temp2 = matrix[matrixLen-i][j]
                matrix[matrixLen-i][j] = temp1
                matrix[j][i] = temp2

# 150 Evaluate Reverse Polish Notation
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        validOperators = ('+', '-', '*', '/')
        evalStack = []
        for token in tokens:
            if token not in validOperators:
                evalStack.append(int(token))
            else:
                operand2 = evalStack.pop()
                operand1 = evalStack.pop()
                if token == '+':
                    evalStack.append(operand1 + operand2)
                elif token == '-':
                    evalStack.append(operand1 - operand2)
                elif token == '*':
                    evalStack.append(operand1 * operand2)
                else:
                    evalStack.append(int(operand1 / operand2))
        return evalStack.pop()

# 200 Number of Islands
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        islandCount = 0
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                currEntry = grid[y][x]
                if currEntry == '1':
                    islandCount += 1
                    grid = clearIsland(x,y,grid)
        return islandCount

def clearIsland(x,y,grid):
    if x >= 0 and x < len(grid[0]) and y >= 0 and y < len(grid) and grid[y][x] == "1":
        grid[y][x] = "0"
        grid = clearIsland(x-1,y,grid)
        grid = clearIsland(x+1,y,grid)
        grid = clearIsland(x,y-1,grid)
        grid = clearIsland(x,y+1,grid)
    return grid

# 215 Kth largest element in an array
import heapq

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        klargest = []
        for num in nums:
            heapq.heappush(klargest, num)
            if (len(klargest) > k):
                heapq.heappop(klargest)
        return klargest[0]
        