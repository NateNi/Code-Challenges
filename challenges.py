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