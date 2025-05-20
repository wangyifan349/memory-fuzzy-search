# 🧐 In-Memory Fuzzy Search Tool

> This is a pure Python fuzzy search utility that helps you quickly find content similar to your search keywords from large amounts of texts and code stored in program memory.
It supports two matching algorithms, ignores case sensitivity, and requires no third-party libraries for easy integration 🎉

---

## 🚀 Features

- 🔎 Supports fuzzy searching of **multi-line strings as a whole**  
- 🧩 Supports splitting multi-line strings and searching **line by line** to find the best matching lines  
- 🎯 Two string similarity algorithms available:  
  - **Longest Common Subsequence (LCS)** – balances order and fault tolerance  
  - **Levenshtein Distance** – calculates edit steps with more precise detail  
- 🐍 Fully hand-written Python implementation with zero dependencies for easy and fast integration  
- ⚙️ Supports assigning IDs or tags to data entries for quick result location  
- 📈 Returns **similarity scores** for each match, with threshold filtering  
- 💡 Simple and clear code and algorithms, easy to customize and extend

---

## 📝 How to Use

### 1. Prepare Your Data

Your data must be stored in memory, either as:

- A list of plain text strings  
  ```python
  [ "text1", "code snippet 2", ... ]
  ```  
- Or a list of tuples with ID tags and content  
  ```python
  [ ("id1", "content1"), ("code2", "code string"), ... ]
  ```

### 2. Call the Functions

- `fuzzy_search_in_memory(data, query, threshold=0.7, method="lcs")`  
  Performs fuzzy matching on each multi-line string **as a whole**, suitable for large code blocks or articles.  
- `fuzzy_search_multiline_lines(data, query, threshold=0.7, method="lcs")`  
  Splits multi-line strings and **matches line by line** to find the most relevant lines, convenient for precise location.

### 3. Parameter Explanation

| Parameter   | Description                            | Example      |
| ----------- | ----------------------------------   | ----------- |
| `data`      | List of strings or tuples (ID, content) | `[ "text", ("id", "code") ]` |
| `query`     | Search keyword string                  | `"my_function"` |
| `threshold` | Filtering threshold (0 to 1), minimum similarity to keep | `0.7`       |
| `method`    | Algorithm choice: `"lcs"` or `"levenshtein"` | `"lcs"`     |

### 4. Return Values

- **Whole string search:** Returns `(ID or index, content string, similarity score)` for each match  
- **Line-by-line search:** Returns `(ID or index, line number, line content, similarity score)` for each matching line

---

## 💻 Example Code

```python
dataset = [
    "def my_function(x):\n    print(x)\n    return x * 2",
    ("code1", "def my_funtion(y):\n    print(y)\n    return y + 1"),
    "This is a test text\ncontaining multiple lines"
]

query = "def my_function"

# Whole string fuzzy search
results = fuzzy_search_in_memory(dataset, query, threshold=0.6, method="lcs")
for id_, text, score in results:
    print(f"🔎 Match ID/Index: {id_}, Similarity: {score:.2f}")
    print(text)
    print("---")

# Line-by-line fuzzy search
results_lines = fuzzy_search_multiline_lines(dataset, query, threshold=0.6, method="levenshtein")
for id_, line_no, line_text, score in results_lines:
    print(f"📝 Match ID/Index: {id_}, Line: {line_no}, Similarity: {score:.2f}")
    print(line_text)
    print("---")
```

---

## ⚠️ Tips

- Both search keywords and data are automatically converted to **lowercase** to ensure case-insensitive matching  
- Whole text search for long content can be computationally heavy; prefer line-by-line searching for better speed and accurate pinpointing  
- Threshold values are typically set between 0.5 and 0.9; higher values mean stricter matches, lower values allow looser matches  
- Sorting uses simple bubble sort suitable for small to medium data sizes; consider optimizing for large-scale data

---

## 📦 Dependencies

- Pure Python 3 code, no third-party dependencies or library installations required

---

## 🥳 Feedback Welcome

Feel free to share any suggestions or feature requests to make this tool even more useful!  
Wish you smooth coding and zero bugs! 🚀
