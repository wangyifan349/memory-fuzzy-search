# Algorithm Implementations in Python

This repository contains Python implementations of various algorithms, including similarity measures, distance metrics, string algorithms, sequence problems, search algorithms, dimensionality reduction techniques, and text vectorization methods. 
The implementations cover mathematical formulations, computational logic, and practical usages.

---

## Table of Contents

1. [Similarity and Distance Metrics](#1-similarity-and-distance-metrics)
   - [1.1 Cosine Similarity](#11-cosine-similarity)
   - [1.2 Dot Product](#12-dot-product)
   - [1.3 Euclidean Distance](#13-euclidean-distance)
   - [1.4 Manhattan Distance](#14-manhattan-distance)
   - [1.5 Mahalanobis Distance](#15-mahalanobis-distance)
   - [1.6 Jaccard Similarity](#16-jaccard-similarity)
   - [1.7 Hamming Distance](#17-hamming-distance)
   - [1.8 Vector Normalization](#18-vector-normalization)
2. [String Algorithms](#2-string-algorithms)
   - [2.1 Longest Common Subsequence (LCS)](#21-longest-common-subsequence-lcs)
   - [2.2 Edit Distance (Levenshtein Distance)](#22-edit-distance-levenshtein-distance)
3. [Fibonacci Sequence](#3-fibonacci-sequence)
   - [3.1 Recursive Version](#31-recursive-version)
   - [3.2 Dynamic Programming Version](#32-dynamic-programming-version)
4. [Knuth-Morris-Pratt (KMP) Algorithm](#4-knuth-morris-pratt-kmp-algorithm)
   - [4.1 Prefix Table Computation](#41-prefix-table-computation)
   - [4.2 KMP String Matching](#42-kmp-string-matching)
5. [Dimensionality Reduction Techniques](#5-dimensionality-reduction-techniques)
   - [5.1 Principal Component Analysis (PCA)](#51-principal-component-analysis-pca)
   - [5.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)](#52-t-distributed-stochastic-neighbor-embedding-t-sne)
   - [5.3 Uniform Manifold Approximation and Projection (UMAP)](#53-uniform-manifold-approximation-and-projection-umap)
6. [Text Vectorization and Word Embeddings](#6-text-vectorization-and-word-embeddings)
   - [6.1 TF-IDF (Term Frequency-Inverse Document Frequency)](#61-tf-idf-term-frequency-inverse-document-frequency)
   - [6.2 Bag of Words (BoW)](#62-bag-of-words-bow)
   - [6.3 Word2Vec Training](#63-word2vec-training)

---

## 1. Similarity and Distance Metrics

### 1.1 Cosine Similarity

#### **Introduction**

Cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space. It is a measure of orientation and not magnitude.

#### **Mathematical Formulation**

For two vectors **A** and **B**, the cosine similarity is given by:

\[
\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \times \|\mathbf{B}\|_2}
\]

Where:

- \(\mathbf{A} \cdot \mathbf{B}\) is the dot product of **A** and **B**.
- \(\|\mathbf{A}\|_2\) is the Euclidean norm (L2 norm) of vector **A**.

#### **Computational Logic**

1. Compute the dot product of the two vectors.
2. Compute the L2 norms of each vector.
3. Divide the dot product by the product of the norms.
4. Handle the case when any vector norm is zero.

#### **Usage and Applications**

- Text similarity in Natural Language Processing.
- Measuring document similarity in Information Retrieval.
- Recommendation systems.

---

### 1.2 Dot Product

#### **Introduction**

The dot product (scalar product) of two vectors is an algebraic operation that takes two equal-length sequences of numbers and returns a single number.

#### **Mathematical Formulation**

For vectors **A** and **B**:

\[
\mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{n} A_i B_i
\]

#### **Computational Logic**

- Multiply corresponding elements of the two vectors.
- Sum all the products.

#### **Usage and Applications**

- Computing projections in geometry.
- Calculating angles between vectors.
- Physics applications involving work and force.

---

### 1.3 Euclidean Distance

#### **Introduction**

Euclidean distance is the "ordinary" straight-line distance between two points in Euclidean space.

#### **Mathematical Formulation**

For two vectors **A** and **B**:

\[
d(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}
\]

#### **Computational Logic**

1. Compute the difference between corresponding elements.
2. Square each difference.
3. Sum all squared differences.
4. Take the square root of the sum.

#### **Usage and Applications**

- Clustering algorithms (e.g., K-Means).
- Distance calculations in geometry.
- Nearest neighbor searches.

---

### 1.4 Manhattan Distance

#### **Introduction**

Manhattan distance is the sum of the absolute differences of their Cartesian coordinates. Also known as L1 norm.

#### **Mathematical Formulation**

For vectors **A** and **B**:

\[
d(\mathbf{A}, \mathbf{B}) = \sum_{i=1}^{n} |A_i - B_i|
\]

#### **Computational Logic**

- Compute the absolute difference between corresponding elements.
- Sum all absolute differences.

#### **Usage and Applications**

- Grid-based pathfinding.
- Simplistic distance metric in machine learning.
- Optimization problems in mathematics.

---

### 1.5 Mahalanobis Distance

#### **Introduction**

Mahalanobis distance measures the distance between a point and a distribution. It accounts for the correlations of the data set.

#### **Mathematical Formulation**

For vectors **A** and **B**, and covariance matrix **Σ**:

\[
d(\mathbf{A}, \mathbf{B}) = \sqrt{(\mathbf{A} - \mathbf{B})^T \Sigma^{-1} (\mathbf{A} - \mathbf{B})}
\]

Where:

- \(\Sigma^{-1}\) is the inverse of the covariance matrix.

#### **Computational Logic**

1. Compute the difference vector \(\mathbf{D} = \mathbf{A} - \mathbf{B}\).
2. Compute the inverse of the covariance matrix.
3. Compute the quadratic form \( \mathbf{D}^T \Sigma^{-1} \mathbf{D} \).
4. Take the square root of the result.

#### **Usage and Applications**

- Multivariate outlier detection.
- Classification algorithms.
- Cluster analysis.

---

### 1.6 Jaccard Similarity

#### **Introduction**

Jaccard similarity measures the similarity between finite sample sets, defined as the size of the intersection divided by the size of the union of the sample sets.

#### **Mathematical Formulation**

For sets **A** and **B**:

\[
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
\]

#### **Computational Logic**

1. Compute the intersection of the two sets.
2. Compute the union of the two sets.
3. Divide the size of the intersection by the size of the union.

#### **Usage and Applications**

- Document similarity.
- Recommender systems.
- Biological taxonomy.

---

### 1.7 Hamming Distance

#### **Introduction**

Hamming distance between two strings of equal length is the number of positions at which the corresponding symbols are different.

#### **Mathematical Formulation**

For strings **S1** and **S2** of length **n**:

\[
d(S1, S2) = \sum_{i=1}^{n} \delta(S1_i, S2_i)
\]

Where:

- \(\delta(S1_i, S2_i) = 0\) if \(S1_i = S2_i\), else 1.

#### **Computational Logic**

- Compare each character position in both strings.
- Increment the count if characters are different.

#### **Usage and Applications**

- Error detection and correction codes.
- Genetic sequence analysis.
- Cryptography.

---

### 1.8 Vector Normalization

#### **Introduction**

Normalization rescales a vector so that its length (norm) is 1. This is often called making a unit vector.

#### **Mathematical Formulation**

For vector **A**:

\[
\hat{\mathbf{A}} = \frac{\mathbf{A}}{\|\mathbf{A}\|_2}
\]

Where:

- \(\|\mathbf{A}\|_2\) is the Euclidean norm.

#### **Computational Logic**

1. Compute the L2 norm of the vector.
2. Divide each element of the vector by the norm.
3. Handle the case when the norm is zero.

#### **Usage and Applications**

- Preparing data for machine learning algorithms.
- Computing direction without magnitude.
- Normalizing feature vectors.

---

## 2. String Algorithms

### 2.1 Longest Common Subsequence (LCS)

#### **Introduction**

LCS problem is to find the longest subsequence common to all sequences in a set of sequences. A subsequence is a sequence that appears in the same relative order but not necessarily contiguous.

#### **Mathematical Formulation**

Given two sequences **X** of length **m**, and **Y** of length **n**:

Define **c(i, j)** as the length of LCS of **X[1..i]** and **Y[1..j]**.

The recursive formula:

1. If \( X_i = Y_j \):

\[
c(i, j) = c(i-1, j-1) + 1
\]

2. Else:

\[
c(i, j) = \max(c(i-1, j), c(i, j-1))
\]

#### **Computational Logic**

- Initialize a (m+1) x (n+1) matrix **dp** with zeros.
- Fill the matrix using the recursive formula.
- The value at **dp[m][n]** will be the length of the LCS.

#### **Usage and Applications**

- Diff tools for comparing files.
- Bioinformatics for DNA sequence analysis.
- Spell checking and correction.

---

### 2.2 Edit Distance (Levenshtein Distance)

#### **Introduction**

Edit distance measures the minimum number of single-character edits (insertions, deletions, substitutions) required to change one string into another.

#### **Mathematical Formulation**

Given strings **S1** and **S2** of length **m** and **n**:

Let **d(i, j)** be the edit distance between **S1[1..i]** and **S2[1..j]**.

The recurrence relation:

1. **Base cases**:

\[
d(0, j) = j \quad \text{for } j=0 \text{ to } n
\]
\[
d(i, 0) = i \quad \text{for } i=0 \text{ to } m
\]

2. **Recurrence**:

\[
d(i, j) = \min \begin{cases}
d(i-1, j) + 1 & \text{(Deletion)} \\
d(i, j-1) + 1 & \text{(Insertion)} \\
d(i-1, j-1) + \text{cost} & \text{(Substitution)}
\end{cases}
\]

Where **cost** is 0 if \(S1_i = S2_j\), else 1.

#### **Computational Logic**

- Initialize a (m+1) x (n+1) matrix **dp**.
- Fill the base cases.
- Compute **dp[i][j]** using the recurrence relation.
- The edit distance is **dp[m][n]**.

#### **Usage and Applications**

- Spell correction.
- DNA sequence alignment.
- Natural Language Processing.

---

## 3. Fibonacci Sequence

### 3.1 Recursive Version

#### **Introduction**

The Fibonacci sequence is a series of numbers where the next number is found by adding up the two numbers before it.

#### **Mathematical Formulation**

\[
F(n) = \begin{cases}
n & \text{if } n = 0 \text{ or } n = 1 \\
F(n-1) + F(n-2) & \text{if } n > 1
\end{cases}
\]

#### **Computational Logic**

- Use a recursive function where each call computes **F(n-1)** and **F(n-2)**.
- Base cases are when **n** is 0 or 1.

#### **Usage and Applications**

- Mathematical puzzles.
- Algorithm analysis.
- Demonstrating recursive programming.

---

### 3.2 Dynamic Programming Version

#### **Introduction**

Dynamic programming optimizes recursive algorithms by storing intermediate results to avoid redundant computations.

#### **Computational Logic**

1. Create an array **fib** of size **n+1**.
2. Initialize **fib[0] = 0**, **fib[1] = 1**.
3. For **i** from 2 to **n**:
   - **fib[i] = fib[i-1] + fib[i-2]**.
4. Return **fib[n]**.

#### **Usage and Applications**

- Efficient computation of Fibonacci numbers for large **n**.
- Understanding memoization.
- Algorithm optimization techniques.

---

## 4. Knuth-Morris-Pratt (KMP) Algorithm

### 4.1 Prefix Table Computation

#### **Introduction**

KMP algorithm uses a prefix table (also known as the failure function) to determine how much to shift the pattern when a mismatch occurs.

#### **Mathematical Formulation**

Let **lps[i]** be the length of the longest proper prefix which is also a suffix for **pattern[0..i]**.

#### **Computational Logic**

1. Initialize **lps[0] = 0**.
2. For **i** from 1 to length of pattern:
   - If **pattern[i] == pattern[length]**:
     - **length += 1**, **lps[i] = length**.
     - **i += 1**.
   - Else if **length != 0**:
     - **length = lps[length - 1]**.
   - Else:
     - **lps[i] = 0**, **i += 1**.

#### **Usage and Applications**

- Preprocessing step for the KMP algorithm.
- Pattern searching in strings.

---

### 4.2 KMP String Matching

#### **Introduction**

KMP algorithm searches for occurrences of a "pattern" within a main "text string" by employing the observation that when a mismatch occurs, the pattern itself embodies sufficient information to determine where the next match could begin.

#### **Computational Logic**

1. Preprocess the pattern to get the **lps** array.
2. Initialize indices **i = 0** (text index), **j = 0** (pattern index).
3. While **i < len(text)**:
   - If **text[i] == pattern[j]**:
     - **i += 1**, **j += 1**.
     - If **j == len(pattern)**:
       - Record the match position (**i - j**).
       - **j = lps[j - 1]**.
   - Else if **j != 0**:
     - **j = lps[j - 1]**.
   - Else:
     - **i += 1**.

#### **Usage and Applications**

- String searching in texts and documents.
- Computational biology for DNA/protein sequence analysis.
- Text editors' find functionality.

---

## 5. Dimensionality Reduction Techniques

### 5.1 Principal Component Analysis (PCA)

#### **Introduction**

PCA is a statistical technique that transforms data to a new coordinate system, reducing dimensionality by projecting data onto principal components where the variance is maximized.

#### **Mathematical Formulation**

Given data matrix **X**:

1. Compute the mean of each feature.
2. Subtract the mean from data to get zero-centered data.
3. Compute the covariance matrix **C**.
4. Compute eigenvalues and eigenvectors of **C**.
5. Select the top **k** eigenvectors (principal components).
6. Project data onto these eigenvectors.

#### **Computational Logic**

- Use Singular Value Decomposition (SVD) or eigen-decomposition.
- Reduce data to **n_components** dimensions.
- Capture as much variance as possible.

#### **Usage and Applications**

- Data compression.
- Noise reduction.
- Visualization of high-dimensional data.

---

### 5.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)

#### **Introduction**

t-SNE is a non-linear dimensionality reduction technique that is particularly well suited for embedding high-dimensional data into a space of two or three dimensions for visualization purposes.

#### **Computational Logic**

1. Compute pairwise similarities in high-dimensional space.
2. Define a probabilistic model that preserves neighbor relationships.
3. Minimize the Kullback-Leibler divergence between the two probability distributions using gradient descent.

#### **Usage and Applications**

- Visualizing clusters in high-dimensional data.
- Exploring data patterns.
- Image processing and recognition.

---

### 5.3 Uniform Manifold Approximation and Projection (UMAP)

#### **Introduction**

UMAP is a non-linear dimensionality reduction technique that is faster than t-SNE and preserves more of the global structure.

#### **Computational Logic**

1. Construct a fuzzy topological representation of the high-dimensional data.
2. Optimize a low-dimensional graph to be as structurally similar as possible.

#### **Usage and Applications**

- General-purpose dimensionality reduction.
- Visualization.
- Preserving both local and global data structure.

---

## 6. Text Vectorization and Word Embeddings

### 6.1 TF-IDF (Term Frequency-Inverse Document Frequency)

#### **Introduction**

TF-IDF is a numerical statistic intended to reflect how important a word is to a document in a collection or corpus.

#### **Mathematical Formulation**

For term **t** in document **d**:

- **Term Frequency (TF)**:

\[
\text{tf}_{t,d} = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
\]

- **Inverse Document Frequency (IDF)**:

\[
\text{idf}_{t} = \log \left( \frac{N}{|\{ d \in D : t \in d \}|} \right)
\]

Where:

- \( N \) is the total number of documents.
- \( |\{ d \in D : t \in d \}| \) is the number of documents where the term **t** appears.

- **TF-IDF**:

\[
\text{tf-idf}_{t,d} = \text{tf}_{t,d} \times \text{idf}_{t}
\]

#### **Computational Logic**

1. Calculate TF for each term in each document.
2. Calculate IDF for each term across the corpus.
3. Multiply TF and IDF to get TF-IDF score.

#### **Usage and Applications**

- Feature extraction in text mining.
- Information retrieval and ranking.
- Document clustering and classification.

---

### 6.2 Bag of Words (BoW)

#### **Introduction**

BoW is a simplifying representation used in natural language processing, where a text is represented as the bag (multiset) of its words.

#### **Computational Logic**

- Tokenize the text into words.
- Build a vocabulary of known words.
- Count the frequency of each word in the document.

#### **Usage and Applications**

- Text classification.
- Document similarity.
- Preprocessing in NLP pipelines.

---

### 6.3 Word2Vec Training

#### **Introduction**

Word2Vec is a group of models that produce word embeddings. These models are shallow, two-layer neural networks trained to reconstruct linguistic contexts of words.

#### **Training Strategies**

1. **Continuous Bag-of-Words (CBOW)**: Predicts the current word from a window of surrounding context words.
2. **Skip-Gram**: Uses the current word to predict the surrounding window of context words.

#### **Computational Logic**

- Process the corpus into sequences of words.
- Train the neural network on these sequences.
- The hidden layer weights become the word vectors.

#### **Usage and Applications**

- Capturing semantic and syntactic word relationships.
- Input for advanced NLP tasks (e.g., sentiment analysis).
- Improving the performance of downstream machine learning models.

---

## Example Usage and Outputs

### TF-IDF Computation

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love machine learning and NLP",
    "Word2Vec captures semantic relationships",
    "Text processing is fun"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print("TF-IDF Feature Names:\n", feature_names)
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())
```

### Word2Vec Training

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

documents = [
    "I love machine learning and NLP",
    "Word2Vec captures semantic relationships",
    "Text processing is fun"
]

processed_sentences = [simple_preprocess(doc) for doc in documents]
w2v_model = Word2Vec(processed_sentences, vector_size=100, window=5, min_count=1)

word = 'machine'
if word in w2v_model.wv:
    print(f"Word2Vec vector for '{word}':\n", w2v_model.wv[word])
else:
    print(f"'{word}' not in the vocabulary.")
```

---

## Conclusion

This collection provides practical implementations of fundamental algorithms, with emphasis on mathematical accuracy and computational logic. It serves as a valuable resource for understanding and applying these algorithms in fields such as machine learning, data science, natural language processing, and more.

---

## References

- [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Mahalanobis Distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)
- [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Longest Common Subsequence Problem](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem)
- [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [t-SNE](https://lvdmaaten.github.io/tsne/)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf)
- [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf)
- [KMP Algorithm](https://en.wikipedia.org/wiki/Knuth-Morris-Pratt_algorithm)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
