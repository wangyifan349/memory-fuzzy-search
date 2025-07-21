"""
pip install flask numpy faiss-cpu sklearn jieba sentence-transformers
"""

from flask import Flask, request, jsonify, render_template_string  # Flask基本模块
from sentence_transformers import SentenceTransformer            # 句向量模型
import jieba                                                     # 中文分词
import numpy as np                                              # 矩阵运算
import faiss                                                    # 向量索引库
from sklearn.feature_extraction.text import CountVectorizer     # 词袋向量
import re                                                       # 正则处理

app = Flask(__name__)                                           # 初始化Flask应用

QA_DICT = {                                                    # 问答集合
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

questions = list(QA_DICT.keys())                               # 问题列表
answers = list(QA_DICT.values())                               # 答案列表

def chinese_tokenizer(text):                                   # 中文分词，jieba切词并空格连接
    return " ".join([w for w in jieba.lcut(text) if w.strip() != ""])

print("加载微软多语言微调模型...")                             # 加载句向量模型
embed_model = SentenceTransformer("microsoft/Multilingual-MiniLM-L12-H384")

print("计算BERT向量并建索引...")                               # 计算句向量（归一化），构建FAISS索引
questions_embeds = embed_model.encode(questions, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
d = questions_embeds.shape[1]                                  # 向量维度
index_bert = faiss.IndexFlatIP(d)                              # 余弦相似度用内积，IndexFlatIP
index_bert.add(questions_embeds)                               # 加入索引

print("生成词袋向量并建索引...")                               # 词袋，先分词后vectorizer提取向量
questions_corpus = [chinese_tokenizer(q) for q in questions]
vectorizer = CountVectorizer()
vectorizer.fit(questions_corpus)
questions_bow = vectorizer.transform(questions_corpus)
questions_bow_dense = questions_bow.toarray().astype('float32')  # 转numpy密集矩阵
norms = np.linalg.norm(questions_bow_dense, axis=1, keepdims=True)  # 归一化避免0除
norms[norms == 0] = 1
questions_bow_normed = questions_bow_dense / norms
dim_bow = questions_bow_normed.shape[1]
index_bow = faiss.IndexFlatIP(dim_bow)
index_bow.add(questions_bow_normed)

def query_bert_vector(question, top_k=1):                      # 查询BERT向量并返回最相似索引和得分
    q_vec = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    sims, ids = index_bert.search(q_vec, top_k)
    return ids[0][0], float(sims[0][0])

def query_bow_vector(question, top_k=1):                       # 查询词袋向量并返回最相似索引和得分
    q_cut = chinese_tokenizer(question)
    q_vec = vectorizer.transform([q_cut]).toarray().astype('float32')
    norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
    norm[norm == 0] = 1
    q_norm = q_vec / norm
    sims, ids = index_bow.search(q_norm, top_k)
    return ids[0][0], float(sims[0][0])

HTML = '''                                                    # 前端HTML+CSS+JS，支持选择算法，聊天气泡，代码高亮复制
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>医学QA聊天机器人</title>
<style>
  body {
    background: #f0f2f5; margin:0; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; font-size:16px; color:#333;
    display: flex; justify-content: center; align-items: center; height: 100vh;
  }
  #chat-container {
    width: 600px; max-height: 90vh; background:white; border-radius: 12px; box-shadow: 0 4px 16px rgb(0 0 0 / 15%);
    display: flex; flex-direction: column; overflow: hidden;
  }
  #header {
    background:#0a74da; padding: 16px; font-size: 24px; font-weight: 700; color: white; position: relative; user-select: none;
  }
  #algorithm-select {
    position: absolute; right: 20px; top: 14px; font-size: 14px; padding: 6px 10px; border-radius: 6px; border: none; cursor: pointer;
  }
  #chat-box {
    padding: 16px; flex: 1; overflow-y: auto; background: #e9ebee;
    display: flex; flex-direction: column; gap: 12px;
  }
  .message {
    max-width: 75%; line-height: 1.5; word-break: break-word; white-space: pre-wrap; position: relative; display: flex;
  }
  .message.user {
    align-self: flex-end; justify-content: flex-end;
  }
  .message.bot {
    align-self: flex-start; justify-content: flex-start;
  }
  .bubble {
    border-radius: 16px; padding: 12px 18px; font-size: 15px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    position: relative; white-space: pre-wrap;
  }
  .message.user .bubble {
    background: #a0e75a; color: #222; border-bottom-right-radius: 6px;
    animation: fadeInRight 0.3s ease forwards;
  }
  .message.bot .bubble {
    background: white; color: #222; border-bottom-left-radius: 6px;
    animation: fadeInLeft 0.3s ease forwards;
  }
  pre {
    background: #282c34; color: #abb2bf; font-size: 14px; font-family: "Fira Code", Consolas, Monaco, "Courier New", monospace;
    border-radius: 10px; padding: 12px 16px; overflow-x: auto; margin:10px 0 0 0; line-height: 1.4;
  }
  .code-block {
    position: relative;
  }
  .copy-btn {
    position: absolute; top: 8px; right: 10px; background: #0a74da;
    border: none; color: white; border-radius: 6px; font-size: 12px;
    padding: 3px 8px; cursor:pointer; opacity: 0.8; user-select:none;
    transition: opacity 0.3s;
  }
  .copy-btn:hover {
    opacity: 1;
  }
  #input-area {
    display: flex; padding: 12px 16px; border-top: 1px solid #ddd; background: #fafafa; align-items: center;
  }
  #user-input {
    flex: 1; font-size: 16px; padding: 10px 16px; border-radius: 24px; border: 1.5px solid #ccc; outline: none;
    transition: border-color 0.3s;
  }
  #user-input:focus {
    border-color: #0a74da;
  }
  #send-btn {
    margin-left: 16px; background: #0a74da; color: white; font-weight: 600;
    padding: 10px 26px; border-radius: 24px; border:none; cursor:pointer;
    transition: background 0.3s;
  }
  #send-btn:disabled {
    background: #7aaeea; cursor: not-allowed;
  }
  #send-btn:hover:not(:disabled) {
    background: #085abd;
  }
  @keyframes fadeInRight {
    0% {opacity: 0; transform: translateX(40px);}
    100% {opacity: 1; transform: translateX(0px);}
  }
  @keyframes fadeInLeft {
    0% {opacity: 0; transform: translateX(-40px);}
    100% {opacity: 1; transform: translateX(0px);}
  }
</style>
</head>
<body>
  <div id="chat-container">
    <div id="header">
      医学QA聊天机器人
      <select id="algorithm-select" title="选择问答算法">
        <option value="bert" selected>微软微调句向量(BERT)</option>
        <option value="bow">词袋 + FAISS</option>
      </select>
    </div>
    <div id="chat-box" aria-live="polite" aria-atomic="false"></div>
    <form id="input-area" onsubmit="return false;">
      <input id="user-input" type="text" autocomplete="off" placeholder="请输入医学相关问题..." aria-label="输入问题" />
      <button id="send-btn" type="button" aria-label="发送问题">发送</button>
    </form>
  </div>
<script>
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const algoSelect = document.getElementById('algorithm-select');
function escapeHtml(text) { return text.replace(/[&<>"']/g, m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m])); }
function parseMessage(text) {
  const regex = /```(\w*)\n([\s\S]*?)```/g;
  let segments = [], lastIndex=0, match;
  while((match=regex.exec(text))!==null){
    if(match.index>lastIndex) segments.push({type:'text',content:text.slice(lastIndex,match.index)});
    segments.push({type:'code',lang:match[1],content:match[2]});
    lastIndex = regex.lastIndex;
  }
  if(lastIndex<text.length) segments.push({type:'text',content:text.slice(lastIndex)});
  return segments;
}
function renderMessage(text) {
  const segments = parseMessage(text);
  return segments.map(seg=>{
    if(seg.type==='text') return `<span>${escapeHtml(seg.content)}</span>`;
    return `<div class="code-block"><pre><button aria-label="复制代码" class="copy-btn">复制</button><code>${escapeHtml(seg.content)}</code></pre></div>`;
  }).join('');
}
function addMessage(user, text) {
  const div=document.createElement("div");
  div.className="message "+user;
  div.innerHTML=`<div class="bubble">${renderMessage(text)}</div>`;
  chatBox.appendChild(div);
  chatBox.scrollTo({top: chatBox.scrollHeight, behavior: 'smooth'});
  div.querySelectorAll('.copy-btn').forEach(btn=>{
    btn.onclick = ()=>{
      const code = btn.nextElementSibling.textContent;
      navigator.clipboard.writeText(code).then(()=>{
        btn.textContent = '复制成功';
        setTimeout(()=>btn.textContent='复制',1600);
      }).catch(()=>{
        btn.textContent = '复制失败';
        setTimeout(()=>btn.textContent='复制',1600);
      });
    };
  });
}
function sendQuestion() {
  const q = userInput.value.trim();
  if(!q) return;
  addMessage('user', q);
  userInput.value = '';
  sendBtn.disabled=true; userInput.disabled=true;
  const algo = algoSelect.value;
  fetch('/query', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q, algorithm:algo})
  }).then(resp=>resp.json())
  .then(data=>{
    addMessage('bot', data.answer+'\n(相似度得分: '+data.score.toFixed(3)+')');
  }).catch(()=>{
    addMessage('bot', '请求失败，请稍后再试。');
  }).finally(()=>{
    sendBtn.disabled=false; userInput.disabled=false; userInput.focus();
  });
}
sendBtn.onclick = sendQuestion;
userInput.onkeydown=e=>{
  if(e.key==='Enter'&&!e.shiftKey){
    sendQuestion();
    e.preventDefault();
  }
};
window.onload=()=>userInput.focus();
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)                         # 首页返回HTML

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json(force=True)
    question = data.get('question', '').strip()
    algorithm = data.get('algorithm', 'bert')
    if not question:
        return jsonify({"answer": "请输入问题。", "score": 0})
    if algorithm == 'bow':
        idx, score = query_bow_vector(question)                # 词袋算法查询
    else:
        idx, score = query_bert_vector(question)               # BERT算法查询
    answer = answers[idx]
    return jsonify({"answer": answer, "score": score})          # 返回答案和相似度分数

if __name__ == '__main__':
    app.run(debug=True)                                         # 启动服务，调试模式
