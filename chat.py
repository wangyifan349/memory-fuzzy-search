"""
pip install flask scikit-learn numpy jieba
"""


from flask import Flask, render_template_string, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import jieba

app = Flask(__name__)

# 医学问答数据字典
QA_DICT = {
    "什么是高血压？": "高血压是指动脉血压持续升高的慢性疾病。",
    "糖尿病有哪些症状？": """糖尿病常见症状包括多饮、多尿、多食和体重下降。
示例代码（Python）：
```python
def check_symptoms(symptoms):
    common = ['多饮', '多尿', '多食', '体重下降']
    for s in symptoms:
        if s in common:
            print(f"发现典型症状: {s}")
```""",
    "如何预防冠心病？": "预防冠心病应控制血脂，戒烟限酒，适量运动，保持健康体重。",
    "什么是脑卒中？": "脑卒中是脑血管突然阻塞或破裂导致的脑功能障碍。",
}

# ---------------------------
# 文本预处理和分词函数
# ---------------------------

def chinese_tokenizer(text):
    """
    使用jieba进行中文分词，返回以空格分隔的词语字符串
    清除空字符串
    """
    seg_list = jieba.lcut(text)  # 精确模式分词
    filtered = []
    for w in seg_list:
        if w.strip() != "":  # 排除空白词
            filtered.append(w)
    return " ".join(filtered)  # 用空格拼接词语方便向量化

def preprocess(text):
    """
    预处理文本，转小写，去标点，去两端空白
    """
    text = text.lower()
    # 定义常见中英文标点符号，用正则去除
    punctuation = r"[，。、！？【】（）《》“”‘’；：．,.!?()\-\"']"
    text = re.sub(punctuation, "", text)  # 替换为空字符串
    text = text.strip()
    return text

# ---------------------------
# 词袋模型 + 余弦相似度计算
# ---------------------------
def bag_of_words_cosine(query, corpus):
    """
    计算query与corpus中每个句子的余弦相似度
    corpus: list[str], 句子列表
    返回相似度列表，数值0~1，越大越相似
    """
    corpus_cut = []
    for text in corpus:
        # 分词
        cut_text = chinese_tokenizer(text)
        corpus_cut.append(cut_text)
    # query分词
    query_cut = chinese_tokenizer(query)

    # 用CountVectorizer向量化
    vectorizer = CountVectorizer()
    # 拟合所有文本，包括语料和查询
    combined_texts = corpus_cut[:]
    combined_texts.append(query_cut)
    vectorizer.fit(combined_texts)

    # 语料向量
    corpus_vectors = []
    for text in corpus_cut:
        vector = vectorizer.transform([text])
        corpus_vectors.append(vector)

    # 查询向量
    query_vector = vectorizer.transform([query_cut])

    similarity_scores = []
    for vec in corpus_vectors:
        # 余弦相似度计算，返回二维数组，两条向量相似度在[0,1]间
        sim = cosine_similarity(query_vector, vec)
        similarity_scores.append(sim[0][0])  # 取第0行第0列
    return similarity_scores

# ---------------------------
# 编辑距离计算 Edit Distance
# ---------------------------

def edit_distance(s1, s2):
    """
    计算两个字符串s1和s2的编辑距离
    动态规划实现
    返回整数距离，越小越相似
    """
    m = len(s1)
    n = len(s2)
    # 初始化二维dp数组，大小(m+1)*(n+1)
    dp = []
    for i in range(m+1):
        dp.append([0]*(n+1))

    # 边界条件，空字符串转化
    for i in range(m+1):
        dp[i][0] = i  # s1前i个字符删除变空
    for j in range(n+1):
        dp[0][j] = j  # 空字符串插入j个字符变s2

    # 状态转移，三种操作：删除，插入，替换
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                cost = 0  # 字符相等，无代价
            else:
                cost = 1  # 字符不等，替换代价1
            dp[i][j] = min(
                dp[i-1][j] + 1,    # 删除s1[i-1]
                dp[i][j-1] + 1,    # 插入s2[j-1]
                dp[i-1][j-1] + cost # 替换或不变
            )
    return dp[m][n]

# ---------------------------
# 最长公共子序列 LCS 长度计算
# ---------------------------

def lcs_length(s1, s2):
    """
    计算两个字符串的最长公共子序列长度，动态规划实现
    返回整数长度，越大相似度越高
    """
    m = len(s1)
    n = len(s2)
    dp = []
    # 初始化表格，尺寸(m+1)*(n+1)，全0
    for i in range(m+1):
        dp.append([0]*(n+1))

    # 计算LCS状态转移
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                # 左和上取较大值
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# ---------------------------
# 综合编辑距离和LCS计算相似度
# ---------------------------

def edit_lcs_similarity(s1, s2):
    """
    先预处理两个字符串，再计算编辑距离和LCS长度
    计算两者对应的相似度后加权得出最终相似度
    权重编辑距离2，LCS 8
    返回0-1浮点数，相似度越大越好
    """
    s1_processed = preprocess(s1)  # 去标点小写
    s2_processed = preprocess(s2)

    ed = edit_distance(s1_processed, s2_processed)  # 编辑距离
    lcs = lcs_length(s1_processed, s2_processed)   # LCS长度

    max_len = max(len(s1_processed), len(s2_processed))
    if max_len == 0:
        return 1.0  # 空字符串相似度最高

    ed_sim = 1 - (ed / max_len)  # 编辑距离相似度，距离越小相似度越大
    lcs_sim = lcs / max_len       # LCS相似度，长度越大越相似
    # 加权平均
    similarity = (2 * ed_sim + 8 * lcs_sim) / 10
    return similarity

# ---------------------------
# 匹配最优答案
# ---------------------------

def find_best_answer(question, algorithm="edit_lcs"):
    """
    给出用户question，根据algorithm计算相似度，
    找到最匹配问题的答案。
    algorithm可取:
      - 'edit_lcs': 使用编辑距离+LCS加权算法
      - 'cosine': 使用词袋向量 + 余弦相似度算法

    返回答案字符串与相似度分数(float)
    """
    questions = list(QA_DICT.keys())
    answers = list(QA_DICT.values())

    if algorithm == "cosine":
        scores = bag_of_words_cosine(question, questions)  # 词袋余弦相似度评分列表
    else:
        scores = []
        for q in questions:
            scores.append(edit_lcs_similarity(question, q))  # 编辑距离+LCS加权评分列表

    max_score = -1
    max_index = -1
    # 找最大评分及其问题索引
    for idx, sc in enumerate(scores):
        if sc > max_score:
            max_score = sc
            max_index = idx

    # 返回对应答案和分数
    if max_index >= 0 and max_index < len(answers):
        matched_answer = answers[max_index]
    else:
        matched_answer = "抱歉，未找到合适答案。"
    return matched_answer, round(max_score, 3)

# ---------------------------
# Flask 路由
# ---------------------------

# 前端页面，使用render_template_string一次性渲染html+css+js
HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>医学知识QA聊天</title>
<style>
  body {
    margin: 0;
    background: #e5ddd5;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    font-size: 18px;
  }
  #chat-container {
    max-width: 720px;
    margin: 30px auto;
    background: #fff;
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    height: 90vh;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
  }
  #header {
    padding: 15px 20px;
    background: #075e54;
    color: white;
    font-weight: bold;
    font-size: 24px;
    text-align: center;
    border-radius: 10px 10px 0 0;
    position: relative;
  }
  /* 右上角选择框容器 */
  #algorithm-container {
    position: absolute;
    top: 12px;
    right: 20px;
  }
  #algorithm-select {
    font-size: 14px;
    padding: 4px 8px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
  }
  #chat-box {
    flex: 1;
    overflow-y: auto;
    padding: 15px;
    background: #ece5dd;
  }
  .message {
    display: flex;
    margin-bottom: 18px;
    max-width: 80%;
    word-wrap: break-word;
    line-height: 1.4;
    font-size: 18px;
    position: relative;
  }
  .message.user {
    justify-content: flex-end;
  }
  .message.bot {
    justify-content: flex-start;
  }
  .bubble {
    padding: 12px 18px;
    border-radius: 20px;
    max-width: 100%;
    white-space: pre-wrap;
    white-space: -moz-pre-wrap; /* Firefox */
    white-space: -pre-wrap; /* Opera <7 */
    white-space: -o-pre-wrap; /* Opera 7 */
    word-wrap: break-word;
  }
  .message.user .bubble {
    background: #dcf8c6;
    color: #111;
    border-bottom-right-radius: 4px;
  }
  .message.bot .bubble {
    background: white;
    color: #333;
    border-bottom-left-radius: 4px;
  }
  #input-area {
    display: flex;
    padding: 10px 15px;
    background: #f0f0f0;
    border-top: 1px solid #ddd;
    align-items: center;
  }
  #user-input {
    flex: 1;
    font-size: 18px;
    padding: 10px 15px;
    border-radius: 20px;
    border: 1px solid #ccc;
    outline: none;
    margin-right: 10px;
  }
  #send-btn {
    background: #25d366;
    border: none;
    color: white;
    font-weight: bold;
    padding: 12px 20px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 18px;
    transition: background 0.3s ease;
  }
  #send-btn:hover {
    background: #128c4a;
  }
  /* 代码块样式 */
  pre {
    position: relative;
    background: #f6f8fa;
    border-radius: 6px;
    padding: 12px 40px 12px 12px;
    font-size: 16px;
    font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  /* 复制按钮样式 */
  .copy-btn {
    position: absolute;
    top: 6px;
    right: 6px;
    background: #1da57a;
    border: none;
    color: white;
    font-weight: bold;
    padding: 2px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    opacity: 0.75;
    transition: opacity 0.3s ease;
    z-index: 10;
  }
  .copy-btn:hover {
    opacity: 1;
  }
</style>
</head>
<body>
<div id="chat-container">
  <div id="header">
    医学知识问答聊天
    <div id="algorithm-container" title="选择相似度算法">
      <select id="algorithm-select">
        <option value="edit_lcs" selected>编辑距离(2) + LCS(8) 加权</option>
        <option value="cosine">词袋向量 + 余弦相似度</option>
      </select>
    </div>
  </div>
  <div id="chat-box"></div>
  <div id="input-area">
    <input type="text" id="user-input" placeholder="请输入问题..." autocomplete="off" />
    <button id="send-btn">发送</button>
  </div>
</div>

<script>
  const chatBox = document.getElementById("chat-box");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const algorithmSelect = document.getElementById("algorithm-select");

  // 转义HTML，防止注入
  function escapeHtml(text){
    return text.replace(/&/g, "&amp;")
               .replace(/</g, "&lt;")
               .replace(/>/g, "&gt;")
               .replace(/"/g, "&quot;")
               .replace(/'/g, "&#039;");
  }

  // 将文本中的代码块```包裹的内容转换成带复制按钮的<pre><code>形式，保留缩进
  function formatMessageWithCode(text){
    // 正则匹配代码块，支持多段代码
    const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
    let parts = [];
    let lastIndex = 0;
    let match;
    while((match = codeBlockRegex.exec(text)) !== null){
      let index = match.index;
      // 普通文本部分
      let normalText = text.substring(lastIndex, index);
      if(normalText.trim() !== ""){
        parts.push({type:"text", content: normalText});
      }
      // 代码块内容
      parts.push({type:"code", lang: match[1], content: match[2]});
      lastIndex = codeBlockRegex.lastIndex;
    }
    // 结尾普通文本
    let tailText = text.substring(lastIndex);
    if(tailText.trim() !== ""){
      parts.push({type:"text", content: tailText});
    }

    // 构造html字符串
    let htmlArr = parts.map(part => {
      if(part.type === "text"){
        return "<span>" + escapeHtml(part.content) + "</span>";
      } else if(part.type === "code"){
        return `
          <pre>
            <button class="copy-btn" title="复制代码">复制</button>
            <code class="language-${escapeHtml(part.lang)}">${escapeHtml(part.content)}</code>
          </pre>
        `;
      }
    });
    return htmlArr.join("");
  }

  // 添加消息到聊天框
  function addMessage(user, text){
    let messageDiv = document.createElement("div");
    messageDiv.classList.add("message");
    messageDiv.classList.add(user);

    // 格式化文本，带代码高亮和复制按钮
    messageDiv.innerHTML = `<div class="bubble">${formatMessageWithCode(text)}</div>`;
    
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    // 绑定复制按钮点击事件
    const copyBtns = messageDiv.querySelectorAll(".copy-btn");
    copyBtns.forEach(btn => {
      btn.addEventListener("click", function(){
        let codeEl = btn.nextElementSibling;
        if(!codeEl) return;
        let codeText = codeEl.textContent;
        navigator.clipboard.writeText(codeText).then(() => {
          btn.textContent = "复制成功";
          setTimeout(() => btn.textContent = "复制", 1500);
        }).catch(() => {
          btn.textContent = "复制失败";
          setTimeout(() => btn.textContent = "复制", 1500);
        });
      });
    });
  }

  // 发送用户问题到后端
  function sendQuestion(){
    let question = userInput.value.trim();
    if(question === ""){
      return;
    }
    addMessage("user", question);
    userInput.value = "";
    sendBtn.disabled = true;
    userInput.disabled = true;

    let alg = algorithmSelect.value;

    let xhr = new XMLHttpRequest();
    xhr.open("POST", "/query", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

    xhr.onreadystatechange = function(){
      if(xhr.readyState === 4){
        sendBtn.disabled = false;
        userInput.disabled = false;
        if(xhr.status === 200){
          try {
            let resp = JSON.parse(xhr.responseText);
            addMessage("bot", resp.answer + "\n(相似度得分: " + resp.score + ")");
          } catch(e) {
            addMessage("bot", "服务器返回异常，请稍后重试。");
          }
        } else {
          addMessage("bot", "请求失败，状态码："+xhr.status);
        }
        userInput.focus();
      }
    };

    let postData = {
      question: question,
      algorithm: alg
    };
    xhr.send(JSON.stringify(postData));
  }

  sendBtn.addEventListener("click", sendQuestion);
  // 支持回车发送
  userInput.addEventListener("keydown", function(e){
    if(e.key === "Enter"){
      sendQuestion();
      e.preventDefault();
    }
  });

  window.onload = function(){
    userInput.focus();
  };
</script>
</body>
</html>
'''

@app.route("/")
def index():
    """
    返回聊天HTML页面
    """
    return render_template_string(HTML)

@app.route("/query", methods=["POST"])
def query():
    """
    AJAX接口，接收JSON格式：
    {
      "question": "用户提问",
      "algorithm": "edit_lcs"或"cosine"
    }
    返回JSON：
    {
      "answer": "回复内容",
      "score": 相似度分数(float)
    }
    """
    data = request.json
    question = data.get("question", "")
    if question == "":
        # 空字符串返回提示
        return jsonify({"answer": "请输入问题。", "score": 0})
    algorithm = data.get("algorithm", "edit_lcs")

    answer, score = find_best_answer(question, algorithm)

    return jsonify({"answer": answer, "score": score})

if __name__ == "__main__":
    # 运行Flask调试模式
    app.run(debug=True)
