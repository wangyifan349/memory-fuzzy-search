# -*- coding: utf-8 -*-
"""
完全纯Python实现，不依赖任何库，支持忽略大小写的模糊搜索
支持最长公共子序列(LCS)和编辑距离(Levenshtein)
"""

from typing import List, Tuple, Union

# ---------------------
# 计算字符串s1和s2的最长公共子序列长度，比较时忽略大小写
def lcs_length(s1: str, s2: str) -> int:
    s1 = s1.lower()
    s2 = s2.lower()
    m = len(s1)
    n = len(s2)
    dp = []
    for i in range(m + 1):
        dp.append([0] * (n + 1))

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                if dp[i][j + 1] > dp[i + 1][j]:
                    dp[i + 1][j + 1] = dp[i][j + 1]
                else:
                    dp[i + 1][j + 1] = dp[i + 1][j]

    return dp[m][n]

# ---------------------
# 计算最长公共子序列相似度（0~1），忽略大小写
def lcs_similarity(s1: str, s2: str) -> float:
    if s1 == "" and s2 == "":
        return 1.0

    lcs_len = lcs_length(s1, s2)
    max_len = len(s1)
    if len(s2) > max_len:
        max_len = len(s2)
    if max_len == 0:
        return 1.0

    return lcs_len / max_len

# ---------------------
# 手写实现编辑距离计算，忽略大小写
def levenshtein_distance(s1: str, s2: str) -> int:
    s1 = s1.lower()
    s2 = s2.lower()
    m = len(s1)
    n = len(s2)

    # 初始化dp数组
    dp = []
    for i in range(m + 1):
        dp.append([0] * (n + 1))

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    return dp[m][n]

# ---------------------
# 计算编辑距离的相似度（0~1），忽略大小写
def levenshtein_similarity(s1: str, s2: str) -> float:
    if s1 == "" and s2 == "":
        return 1.0

    dist = levenshtein_distance(s1, s2)
    max_len = len(s1)
    if len(s2) > max_len:
        max_len = len(s2)
    if max_len == 0:
        return 1.0

    return 1 - dist / max_len

# ---------------------
# 内存整体字符串模糊搜索，忽略大小写
def fuzzy_search_in_memory(
        data: List[Union[str, Tuple[Union[int,str], str]]],
        query: str,
        threshold: float = 0.7,
        method: str = "lcs"
    ) -> List[Tuple[Union[int,str], str, float]]:
    results = []
    query = query.strip()
    idx = 0

    for item in data:
        if isinstance(item, tuple):
            id_ = item[0]
            text = item[1]
        else:
            id_ = idx
            text = item

        if method == "lcs":
            score = lcs_similarity(query, text)
        elif method == "levenshtein":
            score = levenshtein_similarity(query, text)
        else:
            raise ValueError("method 参数仅支持 'lcs' 或 'levenshtein'")

        if score >= threshold:
            results.append((id_, text, score))

        idx += 1

    # 冒泡排序降序
    n = len(results)
    i = 0
    while i < n:
        j = 0
        while j < n - i - 1:
            if results[j][2] < results[j + 1][2]:
                temp = results[j]
                results[j] = results[j + 1]
                results[j + 1] = temp
            j += 1
        i += 1

    return results

# ---------------------
# 内存多行字符串逐行模糊搜索，忽略大小写
def fuzzy_search_multiline_lines(
        data: List[Union[str, Tuple[Union[int,str], str]]],
        query: str,
        threshold: float = 0.7,
        method: str = "lcs"
    ) -> List[Tuple[Union[int,str], int, str, float]]:
    results = []
    query = query.strip()
    idx = 0

    for item in data:
        if isinstance(item, tuple):
            id_ = item[0]
            text = item[1]
        else:
            id_ = idx
            text = item

        lines = text.split('\n')
        lineno = 1

        for line in lines:
            line_strip = line.strip()
            if method == "lcs":
                score = lcs_similarity(query, line_strip)
            elif method == "levenshtein":
                score = levenshtein_similarity(query, line_strip)
            else:
                raise ValueError("method 参数仅支持 'lcs' 或 'levenshtein'")

            if score >= threshold:
                results.append((id_, lineno, line_strip, score))

            lineno += 1

        idx += 1

    # 冒泡排序降序
    n = len(results)
    i = 0
    while i < n:
        j = 0
        while j < n - i -1:
            if results[j][3] < results[j + 1][3]:
                temp = results[j]
                results[j] = results[j + 1]
                results[j + 1] = temp
            j += 1
        i += 1

    return results

# ------------- 下面是测试代码，直接运行该脚本就会执行 --------------

dataset = []
dataset.append("""def my_function(x):
Print(x)
return X * 2
""")
dataset.append("""这是第一段中文文本。
它包含了换行符，
用于测试搜索。""")
dataset.append(("code1", """def my_funtion(y):  # 有个拼写错误
PRINT(y)
return y + 1
"""))
dataset.append(("note1", "测试Fuzzy Search功能，我很高兴帮助你！"))

query = "def my_function"

print("-----整体对比多行字符串，使用LCS匹配（忽略大小写）-----")
matched = fuzzy_search_in_memory(dataset, query, threshold=0.6, method="lcs")
for item in matched:
    print("ID或索引:", item[0], ", 得分:", round(item[2], 3), ", 内容:\n", item[1])

print("\n-----逐行匹配多行字符串，使用LCS匹配（忽略大小写）-----")
matched_lines = fuzzy_search_multiline_lines(dataset, query, threshold=0.6, method="lcs")
for item in matched_lines:
    print("ID或索引:", item[0], ", 行号:", item[1], ", 得分:", round(item[3], 3), ", 行内容:", item[2])

print("\n-----整体对比多行字符串，使用编辑距离匹配（忽略大小写）-----")
matched_lev = fuzzy_search_in_memory(dataset, query, threshold=0.6, method="levenshtein")
for item in matched_lev:
    print("ID或索引:", item[0], ", 得分:", round(item[2], 3), ", 内容:\n", item[1])

print("\n-----逐行匹配多行字符串，使用编辑距离匹配（忽略大小写）-----")
matched_lev_lines = fuzzy_search_multiline_lines(dataset, query, threshold=0.6, method="levenshtein")
for item in matched_lev_lines:
    print("ID或索引:", item[0], ", 行号:", item[1], ", 得分:", round(item[3], 3), ", 行内容:", item[2])
