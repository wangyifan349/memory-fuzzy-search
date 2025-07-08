from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
# 加载社区微调的句子编码器
model = SentenceTransformer('all-MiniLM-L6-v2')
# 生物医学领域的简单QA字典
QA_dict = {
    "What is DNA?": "DNA is a molecule that carries genetic instructions in living organisms.",
    "What causes diabetes?": "Diabetes is caused by high blood sugar due to insulin issues.",
    "How does vaccination work?": "Vaccination trains the immune system to recognize and fight pathogens.",
    "What is a virus?": "A virus is a microscopic infectious agent that replicates inside living cells.",
    "What are antibiotics?": "Antibiotics are drugs that kill or stop the growth of bacteria.",
    "What is the human genome?": "The human genome is the complete set of genetic information in humans.",
    "How do neurons communicate?": "Neurons communicate via electrical and chemical signals.",
}
query = "What do vaccines do?"
# 编码输入查询
query_emb = model.encode(query, convert_to_tensor=True)
# 把所有问题先放进列表
questions = []
for q in QA_dict:
    questions.append(q)
# 批量编码所有问题
questions_emb = model.encode(questions, convert_to_tensor=True)
# 计算余弦相似度
cos_scores = F.cosine_similarity(query_emb.unsqueeze(0), questions_emb, dim=1)
# 打印每个候选问题的相似度
for i in range(len(questions)):
    print("Question:", questions[i])
    print("Similarity score:", cos_scores[i].item())
    print()
# 找出最高相似度的答案
best_score = -1.0
best_answer = "Sorry, I don't know the answer."
for i in range(len(questions)):
    if cos_scores[i] > best_score:
        best_score = cos_scores[i]
        best_answer = QA_dict[questions[i]]
print("Query:", query)
print("Best answer:", best_answer)
print("Best score:", best_score.item())








import  torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# 示例QA数据
qa_dict = {
    "高血压的治疗方法有哪些？": "高血压的治疗包括生活方式改变和药物治疗，常用药包括ACE抑制剂和钙通道阻滞剂。",
    "糖尿病患者饮食应注意什么？": "糖尿病患者应注意减少糖分摄入，规律饮食，适量运动。",
    "Python如何读取文件？": "使用open函数，比如：with open('file.txt', 'r') as f: data = f.read()。",
    "Python怎么处理异常？": "使用try...except语句捕获异常并处理。"
}
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
def encode_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask_expanded, 1)
        summed_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        return mean_pooled.cpu().numpy()
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))
# 预编码QA问题文本，缓存向量
question_texts = list(qa_dict.keys())
qa_vecs = encode_texts(question_texts)
while True:
    user_query = input("请输入问题（或输入 exit 退出）：").strip()
    if user_query.lower() == "exit":
        print("感谢使用，再见！")
        break
    if not user_query:
        print(">>> 请输入有效的问题！\n")
        continue

    # 编码用户输入
    query_vec = encode_texts([user_query])[0]

    # 匹配最佳答案
    max_score = -1.0
    best_answer = "抱歉，未找到相关答案。"
    for idx, (q, a) in enumerate(qa_dict.items()):
        score = cosine_similarity(query_vec, qa_vecs[idx])
        if score > max_score:
            max_score = score
            best_answer = a

    # 打印格式化后的结果
    print("\n==================== 回答 ====================")
    print(f"用户问 : {user_query}")
    print(f"相似度 : {max_score:.3f}")
    print("回答内容 :")
    print(best_answer)
    print("===============================================\n")
