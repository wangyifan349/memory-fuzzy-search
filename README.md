# 🧐 内存中模糊搜索工具

> 这是一个**纯 Python 实现**的模糊搜索小工具，能帮你快速从程序内存中的大量文本和代码里，找到和你搜索关键词相似的内容。  
> 它支持两种匹配算法，忽略大小写，且不依赖任何第三方库，轻松集成 🎉

---

## 🚀 功能亮点

- 🔎 支持对**多行字符串整体**做模糊搜索  
- 🧩 支持把多行字符串拆开，分行**一行行搜索**，找出匹配度最高的行  
- 🎯 两种字符串相似度算法可选：  
  - **最长公共子序列（LCS）** - 适合兼顾顺序和容错  
  - **编辑距离（Levenshtein）** - 计算编辑步骤，细节更精准  
- 🐍 全 Python 纯手写实现，零依赖，方便快速集成  
- ⚙️ 支持给每条数据定义ID或标签，帮你快速定位结果项  
- 📈 返回每条匹配结果的**相似度分数**，可设阈值过滤  
- 💡 算法和代码设计简单易懂，方便后续自定义和扩展

---

## 📝 如何使用

### 1. 准备数据

你的数据存储在内存中，形式可为：

- 纯文本字符串列表  
  ```python
  [ "文本1", "代码内容2", ... ]
  ```  
- 或包含ID标识的元组列表  
  ```python
  [ ("id1", "内容1"), ("code2", "代码字符串"), ... ]
  ```

### 2. 调用函数

- `fuzzy_search_in_memory(data, query, threshold=0.7, method="lcs")`  
  对每条多行字符串**整体进行模糊匹配**，适合匹配大段代码或文章。  
- `fuzzy_search_multiline_lines(data, query, threshold=0.7, method="lcs")`  
  把多行字符串拆行，**逐行匹配**，帮你找到最匹配关键词的行，便于定位具体内容。

### 3. 参数解释

| 参数        | 说明                                  | 示例          |
| --------- | ----------------------------------- | ----------- |
| `data`     | 字符串列表，或含（ID,内容）的元组列表             | `[ "文本", ("id", "代码") ]` |
| `query`    | 搜索关键词（字符串）                       | `"my_function"` |
| `threshold`| 匹配阈值，范围0~1，决定结果保留的最低相似度     | `0.7`       |
| `method`   | 算法选择，`"lcs"` 或 `"levenshtein"`            | `"lcs"`     |

### 4. 返回结果

- **整体搜索**：每条匹配返回 `(ID或索引, 数据内容, 相似度评分)`  
- **逐行搜索**：每个匹配行返回 `(ID或索引, 行号, 行内容, 相似度评分)`

---

## 💻 示例代码

```python
dataset = [
    "def my_function(x):\n    print(x)\n    return x * 2",
    ("code1", "def my_funtion(y):\n    print(y)\n    return y + 1"),
    "这是测试文本\n包含多行内容"
]

query = "def my_function"

# 整体模糊搜索
results = fuzzy_search_in_memory(dataset, query, threshold=0.6, method="lcs")
for id_, text, score in results:
    print(f"🔎 匹配ID/索引：{id_}，相似度：{score:.2f}")
    print(text)
    print("---")

# 逐行拆分匹配
results_lines = fuzzy_search_multiline_lines(dataset, query, threshold=0.6, method="levenshtein")
for id_, line_no, line_text, score in results_lines:
    print(f"📝 匹配ID/索引：{id_}，行号：{line_no}，相似度：{score:.2f}")
    print(line_text)
    print("---")
```

---

## ⚠️ 小贴士

- 搜索时会自动把关键词和数据都**转成小写**，确保忽略大小写影响  
- 较长文本整体搜索计算量大，建议用拆行搜索，提高速度和定位准确度  
- 阈值调整范围0.5~0.9，数字越高匹配越严格，越低匹配更宽松  
- 排序采用简单冒泡法，适合中小规模数据，如要处理海量数据可自行优化  

---

## 📦 依赖环境

- 纯 Python 3 代码，无第三方依赖，无需安装任何库

---

## 🥳 期待你的反馈

欢迎提出任何建议或需求，让这个工具更实用！  
祝你工作顺利，代码无BUG！🚀

emoji 调整为你喜欢的样式。
