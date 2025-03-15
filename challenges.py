# Findings:
# 
# 1. Use library heapq when building heaps and priority queues
# 2. Rotate matrix by flipping rows, tranposing elements
#

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
        
# 2 Add two numbers

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        resultStart = None
        resultCurrent = None
        currNode1 = l1
        currNode2 = l2
        extra = 0
        while currNode1 is not None or currNode2 is not None or extra:
            newSum = (currNode1.val if currNode1 is not None else 0) + (currNode2.val if currNode2 is not None else 0) + extra
            if resultStart is None:
                resultStart = ListNode()
                resultStart.val = newSum % 10
                resultCurrent = resultStart
            else: 
                resultCurrent.next = ListNode()
                resultCurrent = resultCurrent.next
                resultCurrent.val = newSum % 10
            extra = newSum // 10
            if currNode1 is not None:
                currNode1 = currNode1.next
            if currNode2 is not None:
                currNode2 = currNode2.next
        return resultStart

# 274 H-Index

class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations.sort(reverse=True)
        hIndex = 0
        for citation in citations:
            if citation > hIndex:
                hIndex += 1
            else:
                return hIndex
        return hIndex

# 138 Copy List with Random Pointer

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if head is None:
            return None
        copyStart = Node(head.val)
        ogRefs = [head]
        copyRefs = [copyStart]
        ogRefs, copyRefs = traverseAndCopy(copyStart, head, ogRefs, copyRefs)
        traverseAndSetRandom(copyStart, head, ogRefs, copyRefs)
        return copyStart

def traverseAndCopy(currNodeCopy, currNodeOg, ogRefs, copyRefs):
    if currNodeOg.next is None:
        return ogRefs, copyRefs

    currNodeOg = currNodeOg.next
    currNodeCopy.next = Node(currNodeOg.val)
    currNodeCopy = currNodeCopy.next
    
    ogRefs.append(currNodeOg)
    copyRefs.append(currNodeCopy)
    return traverseAndCopy(currNodeCopy, currNodeOg, ogRefs, copyRefs)

def traverseAndSetRandom(currNodeCopy, currNodeOg, ogRefs, copyRefs):
    if currNodeOg is None:
        return True
    if currNodeOg.random is not None:
        currNodeCopy.random = copyRefs[ogRefs.index(currNodeOg.random)]
    currNodeCopy = currNodeCopy.next
    currNodeOg = currNodeOg.next
    return traverseAndSetRandom(currNodeCopy, currNodeOg, ogRefs, copyRefs)

# 77 Combinations

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        combos = []
        self.generateCombo([], combos, k, list(range(1, n+1)))
        return combos

    def generateCombo(self, combo, combos, k, values):
        if len(combo) == k:
            combos.append(combo[:])
            return True
        if len(values) + len(combo) >= k:
            for value in values[:]:            
                combo.append(values.pop())
                self.generateCombo(combo, combos, k, values[:])
                combo.pop()
        return True

# 46 Permutations

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        results = []
        self.buildPerm([], nums, results)
        return results
    
    def buildPerm(self, perm, nums, results):
        if len(nums) == 0:
            results.append(perm[:])
            return True
        for i in range(len(nums)):
            perm.append(nums[i])
            self.buildPerm(perm, (nums[0:i] if i != 0 else []) + nums[i+1:], results)
            perm.pop()
        return True
        
# 39 Combination Sum

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        
        def findCombos(combo, comboSum, combos, candidates, target):
            if comboSum == target:
                combos.append(combo[:])
                return
            elif comboSum > target:
                return
            for i in range(len(candidates)):
                combo.append(candidates[i])
                comboSum += candidates[i]
                findCombos(combo[:], comboSum, combos, candidates[i:], target)
                comboSum -= candidates[i]
                combo.pop()

        combos = []
        candidates = [candidate for candidate in candidates if candidate <= target ]
        findCombos([], 0, combos, candidates[:], target)
        return combos  

# 73 Set Matrix Zeroes

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        rowsWithZeroes = []
        for colIndex in range(len(matrix)):
            currColHasZero = False
            for rowIndex in range(len(matrix[0])):
                if matrix[colIndex][rowIndex] is 0:
                    if not currColHasZero:
                        currColHasZero = True
                        for i in range(rowIndex):
                            matrix[colIndex][i] = 0
                    if rowIndex not in rowsWithZeroes:
                        rowsWithZeroes.append(rowIndex)
                        for j in range(colIndex):
                            matrix[j][rowIndex] = 0
                elif rowIndex in rowsWithZeroes or currColHasZero:
                    matrix[colIndex][rowIndex] = 0

# 67 Add Binary

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        result = []
        remainder = 0
        for i in range(max(len(a), len(b))):
            currA = int(a[len(a) - i - 1]) if i < len(a) else 0
            currB = int(b[len(b) - i - 1]) if i < len(b) else 0
            sum = currA + currB + remainder
            result.append(str(sum % 2))
            remainder = sum // 2
        if remainder:
            result.append('1')
        result.reverse()
        return "".join(result)