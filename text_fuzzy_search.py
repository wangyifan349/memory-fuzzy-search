# -*- coding: utf-8 -*-
"""
Pure Python implementation of fuzzy search with case-insensitive matching,
supporting Longest Common Subsequence (LCS) and Levenshtein distance algorithms.
No external dependencies.
"""

from typing import List, Tuple, Union  # Import type hints

# ---------------------
# Calculate the length of longest common subsequence (LCS) ignoring case
def lcs_length(s1: str, s2: str) -> int:
    s1 = s1.lower()              # Convert s1 to lowercase for case-insensitive comparison
    s2 = s2.lower()              # Convert s2 to lowercase
    m = len(s1)                  # Length of s1
    n = len(s2)                  # Length of s2
    dp = []                      # Initialize 2D DP array
    for i in range(m + 1):       # Create (m+1) rows
        dp.append([0] * (n + 1)) # Each row has (n+1) columns initialized to 0

    for i in range(m):           # For each character in s1
        for j in range(n):       # For each character in s2
            if s1[i] == s2[j]:  # If current characters match
                dp[i + 1][j + 1] = dp[i][j] + 1  # Increase subsequence length
            else:
                # Otherwise, choose max of left or top cell in DP matrix
                if dp[i][j + 1] > dp[i + 1][j]:
                    dp[i + 1][j + 1] = dp[i][j + 1]
                else:
                    dp[i + 1][j + 1] = dp[i + 1][j]

    return dp[m][n]              # Return final LCS length (bottom-right cell)

# ---------------------
# Calculate normalized LCS similarity (0 to 1) ignoring case
def lcs_similarity(s1: str, s2: str) -> float:
    if s1 == "" and s2 == "":    # If both strings are empty, similarity is 1
        return 1.0

    lcs_len = lcs_length(s1, s2) # Compute LCS length
    max_len = max(len(s1), len(s2))  # Get max string length
    if max_len == 0:             # Avoid division by zero
        return 1.0

    return lcs_len / max_len     # Normalized similarity

# ---------------------
# Calculate Levenshtein distance ignoring case
def levenshtein_distance(s1: str, s2: str) -> int:
    s1 = s1.lower()              # Convert s1 to lowercase
    s2 = s2.lower()              # Convert s2 to lowercase
    m = len(s1)                  # Length of s1
    n = len(s2)                  # Length of s2

    dp = []                      # Initialize DP array
    for i in range(m + 1):       # Create rows
        dp.append([0] * (n + 1)) # Columns initialized to 0

    for i in range(m + 1):       # Initialize first column
        dp[i][0] = i
    for j in range(n + 1):       # Initialize first row
        dp[0][j] = j

    for i in range(1, m + 1):    # Compute distances row-wise
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:  # If chars match, cost 0
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Calculate minimum among substitution, insertion, deletion
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    return dp[m][n]              # Return edit distance

# ---------------------
# Calculate normalized Levenshtein similarity (0 to 1), ignoring case
def levenshtein_similarity(s1: str, s2: str) -> float:
    if s1 == "" and s2 == "":    # Both empty strings are identical
        return 1.0

    dist = levenshtein_distance(s1, s2)  # Compute edit distance
    max_len = max(len(s1), len(s2))      # Max string length
    if max_len == 0:                      # Prevent division by zero
        return 1.0

    return 1 - dist / max_len             # Similarity: 1 - normalized distance

# ---------------------
# Perform fuzzy search on in-memory strings (whole string comparison)
def fuzzy_search_in_memory(
        data: List[Union[str, Tuple[Union[int,str], str]]],  # List of strings or (id, string)
        query: str,                                          # Search keyword
        threshold: float = 0.7,                              # Similarity threshold [0,1]
        method: str = "lcs"                                  # Similarity method: 'lcs' or 'levenshtein'
    ) -> List[Tuple[Union[int,str], str, float]]:               # Returns list of (id/index, text, score)
    results = []                      # List to store matched entries
    query = query.strip()             # Strip whitespace from query
    idx = 0                          # Default index if no custom id provided

    for item in data:
        if isinstance(item, tuple):  # If item is tuple, unpack id and text
            id_ = item[0]
            text = item[1]
        else:                       # Otherwise, use index as id
            id_ = idx
            text = item

        if method == "lcs":          # Compute similarity based on LCS
            score = lcs_similarity(query, text)
        elif method == "levenshtein":   # Or compute based on Levenshtein
            score = levenshtein_similarity(query, text)
        else:
            raise ValueError("method parameter only supports 'lcs' or 'levenshtein'")

        if score >= threshold:      # If similarity above threshold, add to results
            results.append((id_, text, score))

        idx += 1                    # Increment index

    # Sort results by similarity score descending using bubble sort
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

    return results                   # Return sorted list

# ---------------------
# Perform fuzzy search line-by-line on multiline strings
def fuzzy_search_multiline_lines(
        data: List[Union[str, Tuple[Union[int,str], str]]],  # List of strings or (id, string)
        query: str,
        threshold: float = 0.7,
        method: str = "lcs"
    ) -> List[Tuple[Union[int,str], int, str, float]]:          # Returns list of (id/index, line num, line text, score)
    results = []                      # List to store matched lines
    query = query.strip()             # Strip whitespace around query
    idx = 0                          # Default index

    for item in data:
        if isinstance(item, tuple):
            id_ = item[0]
            text = item[1]
        else:
            id_ = idx
            text = item

        lines = text.split('\n')     # Split text into lines
        lineno = 1                  # Line number tracker

        for line in lines:
            line_strip = line.strip()
            if method == "lcs":
                score = lcs_similarity(query, line_strip)
            elif method == "levenshtein":
                score = levenshtein_similarity(query, line_strip)
            else:
                raise ValueError("method parameter only supports 'lcs' or 'levenshtein'")

            if score >= threshold:  # Keep lines above threshold
                results.append((id_, lineno, line_strip, score))

            lineno += 1             # Increment line counter

        idx += 1                    # Increment index

    # Bubble sort descending by similarity score
    n = len(results)
    i = 0
    while i < n:
        j = 0
        while j < n - i - 1:
            if results[j][3] < results[j + 1][3]:
                temp = results[j]
                results[j] = results[j + 1]
                results[j + 1] = temp
            j += 1
        i += 1

    return results

# --------------------- Interactive Test Loop ---------------------

# Sample dataset mix of strings and tuples
dataset = []
dataset.append("""def my_function(x):
Print(x)
return X * 2
""")
dataset.append("""This is a sample multi-line text.
It contains line breaks,
used for testing the search functionality.""")
dataset.append(("code1", """def my_funtion(y):  # Contains a typo
PRINT(y)
return y + 1
"""))
dataset.append(("note1", "Testing fuzzy search feature, happy to help!"))

print("Fuzzy search tool started! Type your query and press Enter (type 'exit' to quit).")

while True:
    user_query = input("\nEnter search keyword: ").strip()
    if user_query.lower() == "exit":
        print("Exiting fuzzy search tool. Goodbye!")
        break

    # Choose method ('lcs' or 'levenshtein')
    search_method = "lcs"
    # Choose threshold
    threshold_value = 0.6

    print(f"\nSearching for '{user_query}' with method '{search_method}' and threshold {threshold_value}...\n")

    # Overall fuzzy search results
    overall_results = fuzzy_search_in_memory(dataset, user_query, threshold=threshold_value, method=search_method)
    if overall_results:
        print("Overall matching results:")
        for res in overall_results:
            print(f"ID/Index: {res[0]}, Score: {res[2]:.3f}\nContent:\n{res[1]}\n{'-'*40}")
    else:
        print("No matches found in overall search.")

    # Line by line fuzzy search results
    line_results = fuzzy_search_multiline_lines(dataset, user_query, threshold=threshold_value, method=search_method)
    if line_results:
        print("\nLine-by-line matching results:")
        for res in line_results:
            print(f"ID/Index: {res[0]}, Line: {res[1]}, Score: {res[3]:.3f}\nLine Content: {res[2]}\n{'-'*40}")
    else:
        print("No matches found in line-by-line search.")
