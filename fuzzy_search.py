import math
from collections import Counter
from typing import Dict, Tuple

# 示例QA字典
qa_dict = {
    "高血压的治疗方法有哪些？": "高血压的治疗包括生活方式改变和药物治疗，常用药包括ACE抑制剂和钙通道阻滞剂。",
    "糖尿病患者饮食应注意什么？": "糖尿病患者应注意减少糖分摄入，规律饮食，适量运动。",
    "Python如何读取文件？": "使用open函数，比如：with open('file.txt', 'r') as f: data = f.read()。",
    "Python怎么处理异常？": "使用try...except语句捕获异常并处理。"
}

# 简易词频向量化函数，将文本转为小写分词后的Counter对象
def text_to_vector(text: str) -> Counter:
    # 转小写，去除部分标点符号，简单分词空格切分
    words = text.lower().replace('？', '').replace('，', '').replace('。', '').split()
    # 返回词频统计
    return Counter(words)

# 计算余弦相似度函数，输入两个词频Counter向量
def cosine_similarity(vec1: Counter, vec2: Counter) -> float:
    # 计算两个向量的点积
    intersection = set(vec1) & set(vec2)
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    # 计算两个向量的欧几里得范数
    sum1 = sum([v**2 for v in vec1.values()])
    sum2 = sum([v**2 for v in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    # 避免除零
    if not denominator:
        return 0.0
    else:
        return numerator / denominator

# 计算编辑距离（Levenshtein距离）
def levenshtein_distance(s1: str, s2: str) -> int:
    s1 = s1.lower()
    s2 = s2.lower()
    m, n = len(s1), len(s2)
    # 创建动态规划矩阵
    dp = [[0] * (n+1) for _ in range(m+1)]
    # 初始化第一列（空串s2）
    for i in range(m+1):
        dp[i][0] = i
    # 初始化第一行（空串s1）
    for j in range(n+1):
        dp[0][j] = j
    # 填充DP矩阵
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1]+1, dp[i][j-1]+1, dp[i-1][j]+1)
    return dp[m][n]

# 计算最长公共子序列长度（LCS）
def lcs_length(s1: str, s2: str) -> int:
    s1 = s1.lower()
    s2 = s2.lower()
    m, n = len(s1), len(s2)
    # 创建DP矩阵
    dp = [[0]*(n+1) for _ in range(m+1)]
    # 计算LCS
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

# 组合相似度计算函数，结合LCS和编辑距离，返回0到1的相似度
def combined_similarity(s1: str, s2: str, weight_lcs=0.5, weight_lev=0.5) -> float:
    if not s1 and not s2:
        return 1.0  # 两空字符串相似度1
    # 计算编辑距离
    lev_dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    # 归一化编辑距离转相似度
    lev_sim = 1 - lev_dist / max_len if max_len > 0 else 1.0
    # 计算LCS相似度
    lcs_len = lcs_length(s1, s2)
    lcs_sim = lcs_len / max_len if max_len > 0 else 1.0
    # 返回加权平均相似度
    return weight_lcs * lcs_sim + weight_lev * lev_sim

# 主搜索函数，输入查询，qa字典，匹配方法，返回匹配答案和分数
def search_qa(query: str, qa_data: Dict[str, str], method: str = 'vector') -> Tuple[str, float]:
    max_score = -1
    best_answer = "抱歉，未找到相关答案。"
    query = query.strip()
    if method == 'vector':  # 词频向量+余弦相似度
        query_vec = text_to_vector(query)
        for q, a in qa_data.items():
            q_vec = text_to_vector(q)
            score = cosine_similarity(query_vec, q_vec)
            if score > max_score:
                max_score = score
                best_answer = a
    elif method == 'lcs_lev':  # LCS+编辑距离组合相似度
        for q, a in qa_data.items():
            score = combined_similarity(query, q)
            if score > max_score:
                max_score = score
                best_answer = a
    else:
        raise ValueError("method参数必须是'vector'或'lcs_lev'")
    return best_answer, max_score


if __name__ == "__main__":
    print("欢迎使用QA问答系统，请输入您的问题（输入exit退出）：")
    while True:
        user_query = input("请输入问题：").strip()
        if user_query.lower() == "exit":
            print("感谢使用，再见！")
            break
        if not user_query:
            print("请输入有效的问题！")
            continue

        # 选择匹配方式: 这里示例先用两种都展示
        answer_vec, score_vec = search_qa(user_query, qa_dict, method='vector')
        print(f"[词向量+余弦] 匹配度: {score_vec:.3f}，答案:\n{answer_vec}\n")

        answer_lcs, score_lcs = search_qa(user_query, qa_dict, method='lcs_lev')
        print(f"[LCS+编辑距离] 匹配度: {score_lcs:.3f}，答案:\n{answer_lcs}\n")
