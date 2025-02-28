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