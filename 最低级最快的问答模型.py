# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify

app = Flask(__name__)

def lcs_length(a: str, b: str) -> int:
    """
    计算字符串 a 和 b 的最长公共子序列长度
    动态规划算法，时间复杂度 O(len(a)*len(b))
    """
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def find_best_answer(question: str, qa_pairs: list) -> (str, float):
    """
    在 qa_pairs 中检索与 question LCS 最长的问答对，
    返回 (最佳答案, 相似度分数)
    相似度分数 = LCS_length / max(len(question), len(candidate_question))
    """
    best_score = 0.0
    best_answer = "抱歉，我无法回答该问题。"
    for q, a in qa_pairs:
        l = lcs_length(question, q)
        score = l / max(len(question), len(q))
        if score > best_score:
            best_score = score
            best_answer = a
    return best_answer, best_score

# 预定义医学问答对（示例）
QA_PAIRS = [
    ("什么是高血压？", "高血压是指动脉血压持续升高，收缩压≥140mmHg和/或舒张压≥90mmHg。"),
    ("高血压有哪些症状？", "多数高血压患者早期无明显症状，严重者可头痛、头晕、心悸、耳鸣等。"),
    ("如何预防骨质疏松？", "预防骨质疏松应注意钙和维生素D的摄入、适量运动、戒烟限酒、避免过度减重。"),
    ("糖尿病的常见并发症有哪些？", "糖尿病并发症可分为急性并发症和慢性并发症，慢性包括心血管病变、肾病、视网膜病变、神经病变等。"),
    ("什么是冠心病？", "冠心病是冠状动脉粥样硬化引起心肌缺血、缺氧或坏死的疾病。"),
]

@app.route("/answer", methods=["GET"])
def answer():
    """
    HTTP GET 参数：
      question: 用户的提问
    返回 JSON：
      {
        "question": "...",
        "answer": "...",
        "score": 0.75
      }
    """
    user_q = request.args.get("question", "").strip()
    if not user_q:
        return jsonify({
            "error": "缺少 question 参数或参数为空"
        }), 400

    best_answer, score = find_best_answer(user_q, QA_PAIRS)
    return jsonify({
        "question": user_q,
        "answer": best_answer,
        "score": round(score, 4)
    })

if __name__ == "__main__":
    # 默认监听 0.0.0.0:5000
    app.run(host="0.0.0.0", port=5000)
