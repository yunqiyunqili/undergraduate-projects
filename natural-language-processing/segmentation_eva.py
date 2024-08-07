from collections import Counter

with open('output.txt', 'r', encoding='utf-8') as f:
    # 读取文件中的每一行数据，去掉末尾的换行符，并添加到一个列表中
    test_list = [line.strip() for line in f.readlines()]

with open('groundtruthword.txt', 'r', encoding='utf-8') as f:
    # 读取文件中的每一行数据，去掉末尾的换行符，并添加到一个列表中
    golden_list = [line.strip() for line in f.readlines()]

# 将golden_list和test_list中的每个句子拆分成词语
golden_words = [word for sentence in golden_list for word in sentence.split('|')]
test_words = [word for sentence in test_list for word in sentence.split('|')]

# 统计每个词语在golden_list和test_list中出现的次数
golden_counts = Counter(golden_words)
test_counts = Counter(test_words)

# 计算准确率、召回率和F1值
intersection = set(golden_words) & set(test_words)
precision = sum([test_counts[word] for word in intersection]) / len(test_words)
recall = sum([golden_counts[word] for word in intersection]) / len(golden_words)
f1_score = 2 * precision * recall / (precision + recall)

print('Precision: {:.2f}%'.format(precision * 100))
print('Recall: {:.2f}%'.format(recall * 100))
print('F1 score: {:.2f}%'.format(f1_score * 100))
