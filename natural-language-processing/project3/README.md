## 基于Pytorch 和 PaddlePaddle 实现 命名实体识别（NER）
### PyTorch 实现
    框架：PyTorch
    
    模型：BiLSTM-CRF
    
        BiLSTM（双向 LSTM）：捕捉词语的上下文信息，考虑前后词的关系。
        
        CRF（条件随机场）：优化序列标签的预测，使得预测序列符合标签的转移约束。
        
    特征处理：
    
        词嵌入：通过 nn.Embedding 层处理词汇。
        
        上下文特征：通过双向 LSTM 层处理上下文信息。
### PaddlePaddle 实现
    框架：PaddlePaddle
    
    模型：Stacked LSTM + CRF
    
        Stacked LSTM（栈式 LSTM）：堆叠多个 LSTM 层来捕捉更复杂的序列特征。
        
        CRF（条件随机场）：用于序列标注和优化标签预测。
        
    特征处理：
    
        词嵌入：通过 fluid.embedding 层处理词汇和上下文特征。
        
        上下文特征：包括谓词和上下文区域标志的多维表示。
        
    
