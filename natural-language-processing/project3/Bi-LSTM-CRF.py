#coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import load_sentences,char_mapping,tag_mapping
from sklearn.metrics import f1_score,precision_score, recall_score

torch.manual_seed(1)
START_TAG, END_TAG = "<s>", "<e>"


def log_sum_exp(smat):

    vmax = smat.max(dim=0, keepdim=True).values
    return (smat - vmax).exp().sum(axis=0, keepdim=True).log() + vmax


class BiLSTM_CRF(nn.Module):
    def __init__(self, tag2ix, word2ix, embedding_dim, hidden_dim):
        """
        :param tag2ix: 序列标注问题的 标签 -> 下标 的映射
        :param word2ix: 输入单词 -> 下标 的映射
        :param embedding_dim: 喂进BiLSTM的词向量的维度
        :param hidden_dim: 期望的BiLSTM输出层维度
        """
        super(BiLSTM_CRF, self).__init__()
        assert hidden_dim % 2 == 0
        self.embedding_dim, self.hidden_dim = embedding_dim, hidden_dim
        self.tag2ix, self.word2ix, self.n_tags = tag2ix, word2ix, len(tag2ix)
        self.word_embeds = nn.Embedding(len(word2ix), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.n_tags)
        self.transitions = nn.Parameter(torch.randn(self.n_tags, self.n_tags))


    def neg_log_likelihood(self, words, tags): #求一对 <sentence, tags> 在当前参数下的负对数似然，作为loss
        frames = self._get_lstm_features(words)
        gold_score = self._score_sentence(frames, tags)
        forward_score = self._forward_alg(frames)
        # -(正确路径的分数 - 所有路径的分数和），注意取负号 -log(a/b) = -[log(a) - log(b)] = log(b) - log(a)
        return forward_score - gold_score

    def _get_lstm_features(self, words):

        embeds = self.word_embeds(self._to_tensor(words, self.word2ix)).view(len(words), 1, -1)
        hidden = torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2)
        lstm_out, _hidden = self.lstm(embeds, hidden)
        return self.hidden2tag(lstm_out.squeeze(1))

    def _score_sentence(self, frames, tags):
        """
        求路径pair: frames->tags 的分值
        index:      0   1   2   3   4   5   6
        frames:     F0  F1  F2  F3  F4
        tags:  <s>  Y0  Y1  Y2  Y3  Y4  <e>
        """
        tags_tensor = self._to_tensor([START_TAG] + tags, self.tag2ix)
        score = torch.zeros(1)
        for i, frame in enumerate(frames):
            score += self.transitions[tags_tensor[i], tags_tensor[i + 1]] + frame[tags_tensor[i + 1]]
        return score + self.transitions[tags_tensor[-1], self.tag2ix[END_TAG]]

    def _forward_alg(self, frames):
        ####给定每一帧的发射分值; 按照当前的CRF层参数算出所有可能序列的分值和，用作概率归一化分母
        alpha = torch.full((1, self.n_tags), -10000.0)
        alpha[0][self.tag2ix[START_TAG]] = 0
        for frame in frames:
            # log_sum_exp()相加: 当前各状态的分值分布(列向量) + 发射分值(行向量) + 转移矩阵(方形矩阵)，然后按列求log_sum_exp得到行向量
            alpha = log_sum_exp(alpha.T + frame.unsqueeze(0) + self.transitions)
        return log_sum_exp(alpha.T + 0 + self.transitions[:, [self.tag2ix[END_TAG]]]).flatten()

    def _viterbi_decode(self, frames):
        backtrace = []
        alpha = torch.full((1, self.n_tags), -10000.)
        alpha[0][self.tag2ix[START_TAG]] = 0
        for frame in frames:

            smat = alpha.T + frame.unsqueeze(0) + self.transitions
            backtrace.append(smat.argmax(0))
            alpha = log_sum_exp(smat)

        # 回溯路径
        smat = alpha.T + 0 + self.transitions[:, [self.tag2ix[END_TAG]]]
        best_tag_id = smat.flatten().argmax().item()
        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)

        return log_sum_exp(smat).item(), best_path[::-1]  ##返回最优路径分值 和 最优路径

    def forward(self, words):
        lstm_feats = self._get_lstm_features(words)
        return self._viterbi_decode(lstm_feats)

    def _to_tensor(self, words, to_ix):
        return torch.tensor([to_ix[w] for w in words], dtype=torch.long)



###对训练数据进行格式转换#####
def datatrans(train_sentences):
    sentence = []
    for sen in train_sentences:
        words_list = []
        tag_list = []
        for item in sen:
            words_list.append(item[0])
            tag_list.append(item[1])
        sentence.append((words_list,tag_list))
    return sentence


def eval(model, dev_data, id_to_tag):
    pred_res = []
    gold_res = []
    for words, tags in dev_data:
        best_path_score, best_path = model.forward(words)
        pred = ([id_to_tag[int(x)] for x in best_path])
        pred_res.extend(pred)
        gold_res.extend(tags)

    f1 = f1_score(gold_res, pred_res, average='micro')
    precison = precision_score(gold_res, pred_res, average='micro')
    recall = recall_score(gold_res, pred_res, average='micro')
    print("f1:%.4f, precision:%.4f, recall:%.4f" % (f1, precison, recall))


'''

if __name__ == "__main__":
    training_data = [("the wall street journal reported today that apple corporation made money".split(),
                      "B I I I O O O B I O O".split()),
                     ("georgia tech is a university in georgia".split(), "B I O O O O B".split())]

    model = BiLSTM_CRF(tag2ix={"B": 0, "I": 1, "O": 2, START_TAG: 3, END_TAG: 4},
                       word2ix={w: i for i, w in enumerate({w for s, _ in training_data for w in s})},
                       embedding_dim=5, hidden_dim=4)

    # with torch.no_grad():  # 训练前, 观察一下预测结果(应该是随机或者全零参数导致的结果)
    #     print(model(training_data[0][0]))

    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    for epoch in range(1):  # 不要试图改成100, 在这个教学例子数据集上会欠拟合……
        for words, tags in training_data:
            model.zero_grad()  # PyTorch默认会累积梯度; 而我们需要每条样本单独算梯度
            model.neg_log_likelihood(words, tags).backward()  # 前向求出负对数似然(loss); 然后回传梯度
            optimizer.step()  # 梯度下降，更新参数

    # 训练后的预测结果(有意义的结果，与label一致); 打印类似 (18.722553253173828, [0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
    with torch.no_grad():  # 这里用了第一条训练数据(而非专门的测试数据)，仅作教学演示
        print(model(training_data[0][0]))
'''
def main():
    train_sentences = load_sentences('data/example.train')
    dev_sentences = load_sentences('data/example.test')
    train_data = datatrans(train_sentences)
    dev_data = datatrans(dev_sentences)
    #for item in train_sentences:
        #print(item)
    tag_to_id , id_to_tag = tag_mapping(train_sentences)
    tag_to_id.update({START_TAG:6,END_TAG:7})
    char_to_id,id_to_char = char_mapping(train_sentences)
    embedding_dim = 64
    hidden_dim = 32
    model = BiLSTM_CRF(tag2ix=tag_to_id,
                       word2ix=char_to_id,
                       embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    for chars,tags in train_data:
        model.zero_grad()  # PyTorch默认会累积梯度; 而我们需要每条样本单独算梯度
        model.neg_log_likelihood(chars, tags).backward()  # 前向求出负对数似然(loss); 然后回传梯度
        optimizer.step()  # 梯度下降，更新参数
    eval(model,dev_data,id_to_tag)



if __name__=='__main__':
    main()
