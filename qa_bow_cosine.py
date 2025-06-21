import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# QA corresponding example, supports mixed Chinese and English
qa_dict = {
    "What is machine learning?": "Machine learning is a branch of artificial intelligence that refers to methods where computers learn from data and improve performance.",
    "How to improve Python code performance?": "Use profiling tools to identify bottlenecks, leverage built-in functions, and consider algorithms optimization.",
    "What holiday is Labor Day?": "Labor Day is a holiday for international workers, aimed at commemorating the contributions of workers.",
    "Explain the concept of blockchain.": "Blockchain is a decentralized ledger technology that ensures data integrity and transparency across networks.",
    "How to prevent phishing attacks?": "To prevent phishing attacks, pay attention to the authenticity of email links, enable two-factor authentication, and keep security software updated."
}

# Jieba tokenizer for CountVectorizer
def jieba_tokenizer(text):
    return list(jieba.cut(text))

# Construct CountVectorizer, passing in Jieba segmentation for tokenizer, default counts only term frequency (TF)
vectorizer = CountVectorizer(tokenizer=jieba_tokenizer)

# Build corpus (list of questions)
questions = list(qa_dict.keys())

# Train bag-of-words term frequency
X = vectorizer.fit_transform(questions)  # shape = (num_questions, vocab_size)

def tf_cosine_match(query, vectorizer, X, questions, qa_dict):
    # Vectorize the query in the same way
    q_vec = vectorizer.transform([query])  # shape = (1, vocab_size)
    
    # Calculate cosine similarity between query and all questions
    cosine_similarities = cosine_similarity(q_vec, X).flatten()  # shape = (num_questions,)
    
    # Find the highest similarity index
    best_idx = np.argmax(cosine_similarities)
    best_score = cosine_similarities[best_idx]
    
    # If the similarity is too low, consider returning a prompt for no suitable answer
    if best_score < 0.1:
        return "Sorry, no relevant answer found.", best_score
    
    best_question = questions[best_idx]
    best_answer = qa_dict[best_question]
    return best_answer, best_score

if __name__ == "__main__":
    print("Welcome to the intelligent Q&A system based on TF bag-of-words + cosine similarity!")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nPlease enter your question:").strip()
        if query.lower() == 'exit':
            print("Thank you for using, goodbye!")
            break
        if not query:
            print("Please enter a valid question!")
            continue
        
        answer, score = tf_cosine_match(query, vectorizer, X, questions, qa_dict)
        print(f"Match degree: {score:.4f}\nAnswer: {answer}")
