from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 加载预训练的词向量模型
word_vectors = {}  # 存储词向量的字典

# 读取词向量文件
with open('/Users/palekiller/Downloads/glove.6B/glove.6B.200d.txt', 'r', encoding='utf-8') as file:
    for line in file:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:], dtype=np.float32)
        word_vectors[word] = vector

# 计算词语相似度
word1 = 'apple'
word2 = 'banana'

if word1 in word_vectors and word2 in word_vectors:
    vector1 = word_vectors[word1].reshape(1, -1)
    vector2 = word_vectors[word2].reshape(1, -1)
    similarity = cosine_similarity(vector1, vector2)[0][0]
    print("相似度:", similarity)
else:
    print("无法找到词向量")
