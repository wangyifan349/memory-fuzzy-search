from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import jieba

app = Flask(__name__)

# 医学QA数据，题目-答案字典
QA_DICT = {
    "什么是高血压？": "高血压是指动脉血压持续升高的慢性疾病。",
    "糖尿病有哪些症状？": "糖尿病常见症状包括多饮、多尿、多食和体重下降。",
    "如何预防冠心病？": "预防冠心病应控制血脂，戒烟限酒，适量运动，保持健康体重。",
    "什么是脑卒中？": "脑卒中是脑血管突然阻塞或破裂导致的脑功能障碍。",
}

# 对中文句子做分词处理，返回用空格连接的词串，方便词袋向量处理
def chinese_tokenizer(text):
    # 使用jieba分词
    seg_list = jieba.lcut(text)
    # 过滤空格或换行符之类的
    filtered = []
    for w in seg_list:
        if w.strip() != "":
            filtered.append(w)
    # 用空格连接分词结果
    return " ".join(filtered)

# 对文本做基础预处理，去除标点，转换小写等（用于编辑距离算法）
def preprocess(text):
    text = text.lower()
    # 去除中文及英文标点符号，包括全角符号
    punctuation = r"[，。、！？【】（）《》“”‘’；：．,.!?()\-\"']"
    text = re.sub(punctuation, "", text)
    text = text.strip()
    return text

# 词袋+余弦相似度算法，中文分词后向量化计算相似度
def bag_of_words_cosine(query, corpus):
    # 先分词，并组成新的语料库列表
    corpus_cut = []
    for text in corpus:
        cut_text = chinese_tokenizer(text)
        corpus_cut.append(cut_text)
    query_cut = chinese_tokenizer(query)

    # 初始化向量器，拟合所有语料和查询
    vectorizer = CountVectorizer()
    # 先fit所有文本（包含query）
    combined_texts = corpus_cut[:]
    combined_texts.append(query_cut)
    vectorizer.fit(combined_texts)

    # 转换语料为向量
    corpus_vectors = []
    for text in corpus_cut:
        vector = vectorizer.transform([text])
        corpus_vectors.append(vector)

    # 转换query为向量
    query_vector = vectorizer.transform([query_cut])

    # 计算相似度，逐条
    similarity_scores = []
    for vec in corpus_vectors:
        sim = cosine_similarity(query_vector, vec)
        similarity_scores.append(sim[0][0])  # 取二维数组里单个值

    return similarity_scores

# 计算两个字符串的编辑距离（动态规划）
def edit_distance(s1, s2):
    m = len(s1)
    n = len(s2)
    dp = []
    for i in range(m+1):
        dp.append([0]*(n+1))

    # 初始化边界状态
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    # 计算dp值
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[m][n]

# 计算两个字符串的最长公共子序列长度（动态规划）
def lcs_length(s1, s2):
    m = len(s1)
    n = len(s2)
    dp = []
    for i in range(m+1):
        dp.append([0]*(n+1))

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                val1 = dp[i-1][j]
                val2 = dp[i][j-1]
                if val1 > val2:
                    dp[i][j] = val1
                else:
                    dp[i][j] = val2
    return dp[m][n]

# 编辑距离+LCS加权计算相似度，返回分数越大越相似
def edit_lcs_similarity(s1, s2):
    # 预处理，去除标点，转小写
    s1_processed = preprocess(s1)
    s2_processed = preprocess(s2)

    ed = edit_distance(s1_processed, s2_processed)
    lcs = lcs_length(s1_processed, s2_processed)

    max_len = len(s1_processed)
    len2 = len(s2_processed)
    if max_len < len2:
        max_len = len2

    if max_len == 0:
        return 1.0

    ed_sim = 1 - (ed / max_len)
    lcs_sim = lcs / max_len

    similarity = 0.6 * lcs_sim + 0.4 * ed_sim
    return similarity

@app.route("/")
def index():
    # 返回聊天界面html
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    # API接口，接收json请求
    data = request.json
    question = data.get("question", "")
    if question == "":
        return jsonify({"answer": "请输入问题。", "score": 0})

    # 选择算法参数
    algorithm = data.get("algorithm", "cosine")

    questions = []
    answers = []
    # 将QA字典拆成两个列表
    for k in QA_DICT.keys():
        questions.append(k)
    for v in QA_DICT.values():
        answers.append(v)

    if algorithm == "cosine":
        # 词袋向量+余弦相似度
        scores = bag_of_words_cosine(question, questions)
        max_score = -1
        max_index = -1
        index = 0
        # 找最大score及索引
        while index < len(scores):
            score = scores[index]
            if score > max_score:
                max_score = score
                max_index = index
            index = index + 1
        # 找到对应答案
        matched_answer = ""
        if max_index >= 0 and max_index < len(answers):
            matched_answer = answers[max_index]
        else:
            matched_answer = "抱歉，未找到合适答案。"
        return jsonify({"answer": matched_answer, "score": round(max_score, 3)})
    elif algorithm == "edit_lcs":
        # 编辑距离+LCS加权相似度
        scores = []
        i = 0
        while i < len(questions):
            sim = edit_lcs_similarity(question, questions[i])
            scores.append(sim)
            i = i + 1
        max_score = -1
        max_index = -1
        idx = 0
        while idx < len(scores):
            score = scores[idx]
            if score > max_score:
                max_score = score
                max_index = idx
            idx = idx + 1
        matched_answer = ""
        if max_index >= 0 and max_index < len(answers):
            matched_answer = answers[max_index]
        else:
            matched_answer = "抱歉，未找到合适答案。"
        return jsonify({"answer": matched_answer, "score": round(max_score, 3)})
    else:
        # 未知算法，返回错误
        return jsonify({"answer": "未知算法参数", "score": 0})

# 开放api接口，方便wget等命令行工具调用
@app.route("/api/query", methods=["POST"])
def api_query():
    # 接收客户端传json，返回json格式结果
    request_data = request.get_json()
    if request_data is None:
        return jsonify({"error": "请求内容不是JSON格式"}), 400

    question = request_data.get("question", "")
    algorithm = request_data.get("algorithm", "cosine")
    if question == "":
        return jsonify({"answer": "请输入问题。", "score": 0})

    # 和上面一致查询逻辑
    questions = []
    answers = []
    for k in QA_DICT.keys():
        questions.append(k)
    for v in QA_DICT.values():
        answers.append(v)

    if algorithm == "cosine":
        scores = bag_of_words_cosine(question, questions)
        max_score = -1
        max_index = -1
        idx = 0
        while idx < len(scores):
            sc = scores[idx]
            if sc > max_score:
                max_score = sc
                max_index = idx
            idx = idx + 1
        matched_answer = ""
        if max_index >= 0 and max_index < len(answers):
            matched_answer = answers[max_index]
        else:
            matched_answer = "抱歉，未找到合适答案。"
        return jsonify({"answer": matched_answer, "score": round(max_score, 3)})
    elif algorithm == "edit_lcs":
        scores = []
        i = 0
        while i < len(questions):
            sim = edit_lcs_similarity(question, questions[i])
            scores.append(sim)
            i = i + 1
        max_score = -1
        max_index = -1
        idx = 0
        while idx < len(scores):
            sc = scores[idx]
            if sc > max_score:
                max_score = sc
                max_index = idx
            idx = idx + 1
        matched_answer = ""
        if max_index >= 0 and max_index < len(answers):
            matched_answer = answers[max_index]
        else:
            matched_answer = "抱歉，未找到合适答案。"
        return jsonify({"answer": matched_answer, "score": round(max_score, 3)})
    else:
        return jsonify({"answer": "未知算法参数", "score": 0})


if __name__ == "__main__":
    # 启动flask app, debug模式
    app.run(debug=True)





<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>QA聊天</title>
<style>
  /* 页面背景和字体 */
  body {
    margin: 0;
    background: #e5ddd5;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    font-size: 18px;
  }
  /* 主聊天容器，宽度较大，居中 */
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
  /* 顶部标题栏 */
  #header {
    padding: 15px 20px;
    background: #075e54;
    color: white;
    font-weight: bold;
    font-size: 24px;
    text-align: center;
    border-radius: 10px 10px 0 0;
  }
  /* 聊天消息列表，自动滚动，高度灵活 */
  #chat-box {
    flex: 1;
    overflow-y: auto;
    padding: 15px;
    background: #ece5dd;
  }
  /* 每条消息容器 */
  .message {
    display: flex;
    margin-bottom: 18px;
    max-width: 80%;
    word-wrap: break-word;
    line-height: 1.4;
    font-size: 18px;
  }
  /* 用户消息靠右 */
  .message.user {
    justify-content: flex-end;
  }
  /* 系统消息靠左 */
  .message.bot {
    justify-content: flex-start;
  }
  /* 气泡样式 */
  .bubble {
    padding: 12px 18px;
    border-radius: 20px;
    max-width: 100%;
    white-space: pre-wrap;
  }
  /* 用户气泡绿色背景，白字 */
  .message.user .bubble {
    background: #dcf8c6;
    color: #111;
    border-bottom-right-radius: 4px;
  }
  /* 机器人气泡白底黑字 */
  .message.bot .bubble {
    background: white;
    color: #333;
    border-bottom-left-radius: 4px;
  }
  /* 输入框区域 */
  #input-area {
    display: flex;
    padding: 10px 15px;
    background: #f0f0f0;
    border-top: 1px solid #ddd;
    align-items: center;
  }
  /* 文本输入框 */
  #user-input {
    flex: 1;
    font-size: 18px;
    padding: 10px 15px;
    border-radius: 20px;
    border: 1px solid #ccc;
    outline: none;
    margin-right: 10px;
  }
  /* 发送按钮 */
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
  /* 算法选择框 */
  #algorithm-select {
    width: 180px;
    margin: 10px auto 0 auto;
    font-size: 16px;
    display: block;
    padding: 6px 10px;
    border-radius: 8px;
    border: 1px solid #ccc;
  }
</style>
</head>
<body>
<div id="chat-container">
  <div id="header">医学知识问答聊天</div>
  <div id="chat-box">
    <!-- 聊天消息动态插入 -->
  </div>
  <select id="algorithm-select" title="选择相似度算法">
    <option value="cosine">词袋向量 + 余弦相似度</option>
    <option value="edit_lcs">编辑距离 + 最长公共子序列加权</option>
  </select>
  <div id="input-area">
    <input type="text" id="user-input" placeholder="请输入问题..." autocomplete="off" />
    <button id="send-btn">发送</button>
  </div>
</div>

<script>
  // 获取页面元素
  var chatBox = document.getElementById("chat-box");
  var userInput = document.getElementById("user-input");
  var sendBtn = document.getElementById("send-btn");
  var algorithmSelect = document.getElementById("algorithm-select");

  // 向聊天窗口添加消息，user表示身份(user或者bot)，text为内容
  function addMessage(user, text){
    var messageDiv = document.createElement("div");
    messageDiv.classList.add("message");
    if(user === "user"){
      messageDiv.classList.add("user");
    } else {
      messageDiv.classList.add("bot");
    }
    var bubbleDiv = document.createElement("div");
    bubbleDiv.classList.add("bubble");
    bubbleDiv.textContent = text;
    messageDiv.appendChild(bubbleDiv);
    chatBox.appendChild(messageDiv);
    // 滚动到底部
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  // 发送问题请求
  function sendQuestion(){
    var question = userInput.value.trim();
    if(question === ""){
      return;
    }
    // 显示用户消息
    addMessage("user", question);
    userInput.value = "";
    var selectedAlgorithm = algorithmSelect.value;

    // 机器人回复前，禁用发送按钮防止多次提交
    sendBtn.disabled = true;
    userInput.disabled = true;

    // 异步请求后端api
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/query", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

    xhr.onreadystatechange = function(){
      if(xhr.readyState === 4){
        sendBtn.disabled = false;
        userInput.disabled = false;
        if(xhr.status === 200){
          try {
            var response = JSON.parse(xhr.responseText);
            var answer = response.answer;
            var score = response.score;
            var replyText = answer + "\n(相似度得分: " + score + ")";
            addMessage("bot", replyText);
          } catch (e) {
            addMessage("bot", "服务器返回异常，请稍后重试。");
          }
        } else {
          addMessage("bot", "请求失败，状态码：" + xhr.status);
        }
        // 自动聚焦输入框以便继续输入
        userInput.focus();
      }
    };
    // 发送json请求体
    var postData = {
      question: question,
      algorithm: selectedAlgorithm
    };
    xhr.send(JSON.stringify(postData));
  }

  // 绑定按钮点击事件
  sendBtn.addEventListener("click", function(){
    sendQuestion();
  });

  // 绑定回车键事件
  userInput.addEventListener("keydown", function(event){
    if(event.key === "Enter"){
      sendQuestion();
      event.preventDefault();
    }
  });

  // 页面载入时自动聚焦输入框
  window.onload = function(){
    userInput.focus();
  };
</script>
</body>
</html>

wget --post-data='{"question":"什么是高血压？","algorithm":"edit_lcs"}' \
  --header='Content-Type: application/json' \
  -qO- http://127.0.0.1:5000/api/query
