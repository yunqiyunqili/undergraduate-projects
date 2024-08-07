import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch import nn
from model.CNN import singel_cnn
#from model.TRANS import singel_trans
from torchsummary import summary

def criterion(predictions, targets):
    '''
    计算损失。

    :param predictions: 模型的预测值
    :param targets: 目标标签
    :return: 损失值
    '''
    return nn.CrossEntropyLoss()(input=predictions, target=targets)  # 交叉熵损失函数 nn.CrossEntropyLoss() 来计算损失。



def make_train_step(models, criterions, optimizers):
    def train_step(X, Y):
        output_logits, output_softmax = models(X)
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y == predictions) / float(len(Y))
        loss = criterions(output_logits, Y)
        loss.backward()
        optimizers.step()
        optimizers.zero_grad()
        return loss.item(), accuracy * 100

    return train_step



def make_validate_fnc(models, criterions):
    '''
    执行模型的验证。

    :param models: 模型
    :param criterions: 损失函数
    :return: 模型验证函数
    '''
    def validate(X, Y):
        '''
        :param X: 输入数据
        :param Y: 目标标签
        :return: 损失值、准确率和预测结果
        '''
        with torch.no_grad():  # 关闭梯度计算
            models.to('cuda')  # 模型移动到 GPU 上
            models.eval()  # 评估模式
            output_logits, output_softmax = models(X)   # 运行模型，得到输出的logits和softmax结果
            predictions = torch.argmax(output_softmax, dim=1)   # 根据softmax结果，取最大值的索引作为预测结果
            accuracy = torch.sum(Y == predictions) / float(len(Y))  # 计算准确率
            loss = criterions(output_logits, Y)  # 计算损失
        return loss.item(), accuracy * 100, predictions  # 返回损失值、准确率和预测结果

    return validate


def make_save_checkpoint():
    def save_checkpoint(optimizers, models, epoch, filename):
        '''
        保存检查点。

        :param optimizers: 优化器对象
        :param models: 模型对象
        :param epoch: 当前训练的轮数
        :param filename: 检查点保存的文件名
        :return: 无
        '''
        checkpoint_dict = {
            'optimizer': optimizers.state_dict(),
            'model': models.state_dict(),
            'epoch': epoch}
        torch.save(checkpoint_dict, filename)

    return save_checkpoint


def load_checkpoint(optimizer, model, filename):
    '''
    加载检查点。

    :param optimizer: 优化器对象
    :param model: 模型对象
    :param filename: 检查点文件的路径
    :return: 加载的检查点对应的轮数
    '''
    checkpoint_dict = torch.load(filename)  # 加载检查点文件
    epoch = checkpoint_dict['epoch']  # 获取轮数
    model.load_state_dict(checkpoint_dict['model'])  # 将模型的状态字典从检查点中加载到模型对象中
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])  # 将检查点中的优化器状态字典加载到优化器对象中
    return epoch


def train(optimize, mo_del, mini_batch, num_epoch, X_trains, Y_train, X_valids, Y_valid):
    '''
    训练函数。

    :param optimize: 优化器对象
    :param mo_del: 模型对象
    :param mini_batch: mini-batch大小
    :param num_epoch: 训练轮数
    :param X_trains: 训练集输入
    :param Y_train: 训练集标签
    :param X_valids: 验证集输入
    :param Y_valid: 验证集标签
    :return:
    '''
    train_size = X_trains.shape[0]  # 训练集大小
    max_acc = 0  # 最高准确率
    global best_epoch  # 最佳轮数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 检测是否支持GPU加速
    print(f'{device} selected')  # 打印选择的设备
    print('Number of trainable params: ', sum(p.numel() for p in mo_del.parameters()))  # 打印模型的可训练参数数量

    save_checkpoint = make_save_checkpoint()  # 创建保存模型检查点的函数
    train_step = make_train_step(mo_del, criterion, optimizers=optimize)  # 创建训练步骤函数
    validate = make_validate_fnc(mo_del, criterion)  # 创建验证函数

    for epoch in range(num_epoch):  # 循环训练轮数
        mo_del.train()  # 设置模型为训练模式
        ind = np.random.permutation(train_size)  # 随机打乱训练集索引
        X_trains = X_trains[ind, :, :, :]  # 根据打乱的索引重新排序训练集输入
        Y_train = Y_train[ind]  # 根据打乱的索引重新排序训练集标签
        epoch_acc = 0  # 当前轮的累计准确率
        epoch_loss = 0  # 当前轮的累计损失
        num_iterations = int(train_size / mini_batch)  # 计算每轮的迭代次数

        for i in range(num_iterations):  # 循环每轮的迭代次数
            batch_start = i * mini_batch  # 当前迭代的起始索引
            batch_end = min(batch_start + mini_batch, train_size)  # 当前迭代的结束索引
            actual_batch_size = batch_end - batch_start  # 当前迭代的实际批次大小
            X = X_trains[batch_start:batch_end, :, :, :]  # 当前迭代的输入批次
            Y = Y_train[batch_start:batch_end]  # 当前迭代的标签批次
            X_tensor = torch.tensor(X, device=device).float()  # 将输入转换为张量并移到设备上
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)  # 将标签转换为张量并移到设备上
            loss, acc = train_step(X_tensor, Y_tensor)  # 执行训练步骤，返回损失和准确率
            epoch_acc += acc * actual_batch_size / train_size  # 累计当前轮的准确率
            epoch_loss += loss * actual_batch_size / train_size  # 累计当前轮的损失
            print('\r' + f'Epoch {epoch}: iteration {i}/{num_iterations}', end='')  # 打印当前迭代的进度

        X_valid_tensor = torch.tensor(X_valids, device=device).float()  # 将验证集输入转换为张量并移到设备上
        Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.long, device=device)  # 将验证集标签转换为张量并移到设备上
        valid_loss, valid_acc, _ = validate(X_valid_tensor, Y_valid_tensor)  # 执行验证步骤，返回验证损失、准确率和预测结果
        train_losses.append(epoch_loss)  # 记录训练损失
        valid_losses.append(valid_loss)  # 记录验证损失
        train_accs.append(epoch_acc)  # 记录训练准确率
        valid_accs.append(valid_acc)  # 记录验证准确率

        checkpoint_filename = './checkpoint/single_cnn_FINAL-{:03d}.pkl'.format(epoch)  # 检查点文件名

        if max_acc < valid_acc:  # 如果当前验证准确率超过最高准确率
            max_acc = valid_acc  # 更新最高准确率
            best_epoch = epoch  # 更新最佳轮数
            save_checkpoint(optimize, mo_del, epoch, checkpoint_filename)  # 保存模型检查点
            print('The accuracy has been improved to %.4f，save this model at epoch %d' % (max_acc, epoch))  # 打印准确率提高的消息
        print(
            f'\nEpoch {epoch} --- loss:{epoch_loss:.3f}, Epoch accuracy:{epoch_acc:.2f}%, '
            f'Validation loss:{valid_loss:.3f}, Validation accuracy:{valid_acc:.2f}%')  # 打印当前轮的训练和验证结果


if __name__ == '__main__':
    # step1:加载数据
    featuresFileName = './feature/mel40_feat_emodb.pkl'
    with open(featuresFileName, 'rb') as f:
        features = pickle.load(f)
    X_train = features['X_train']  # 加载训练集特征
    X_valid = features['X_valid']  # 加载验证集特征
    X_test = features['X_test']  # 加载测试集特征
    y_train = features['y_train']  # 加载训练集标签
    y_valid = features['y_valid']  # 加载验证集标签
    y_test = features['y_test']  # 加载测试集标签

    # step2:打印网络结构
    device = 'cuda'  # 设置设备为cuda（如果可用）
    model = singel_cnn(7).to(device)  # 创建单卷积神经网络模型，并将其移动到设备上
    summary(model, input_size=(1, 40, 94))  # 打印模型结构和参数数量信息

    # step3:训练
    train_losses, valid_losses, train_accs, valid_accs = [], [], [], []  # 初始化记录训练和验证损失、准确率的列表
    best_epoch = 0  # 初始化最佳轮数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.8)  # 创建优化器
    minibatch = 32  # 设置小批量大小
    num_epochs = 200  # 设置训练轮数
    train(optimizer, model, minibatch, num_epochs, X_train, y_train, X_valid, y_valid)  # 执行训练

    # 绘制训练和验证损失曲线
    plt.title('Loss')
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.plot(train_losses[:], 'b')  # 绘制训练损失曲线
    plt.plot(valid_losses[:], 'r')  # 绘制验证损失曲线
    plt.legend(['Training loss', 'Validation loss'])
    plt.savefig('./result/loss.jpg')  # 保存损失曲线图像
    plt.show()

    # 绘制训练和验证准确率曲线
    plt.title('ACC')
    plt.ylabel('ACC', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.plot(torch.Tensor(train_accs).cpu(), 'b')  # 绘制训练准确率曲线
    plt.plot(torch.Tensor(valid_accs).cpu(), 'r')  # 绘制验证准确率曲线
    plt.legend(['train_accs', 'valid_accs'])
    plt.savefig('./result/acc.jpg')  # 保存准确率曲线图像
    plt.show()

    # step4:测试
    load_folder = './checkpoint'  # 模型检查点文件夹路径
    if best_epoch < 100:
        epoch_str = f'0{best_epoch}'  # 格式化最佳轮数，保证为两位数
    else:
        epoch_str = str(best_epoch)  # 最佳轮数转换为字符串
    model_name = f'single_cnn_FINAL-{epoch_str}.pkl'  # 模型文件名
    load_path = os.path.join(load_folder, model_name)  # 模型文件路径
    model = singel_cnn(7)  # 创建新的单卷积神经网络模型
    load_checkpoint(optimizer, model, load_path)  # 加载训练过程中最佳模型的参数
    print(f'Loaded model from {load_path}')  # 打印加载模型的路径

    validate = make_validate_fnc(model, criterion)  # 创建验证函数
    X_test_tensor = torch.tensor(X_test, device=device).float()  # 转换测试集特征为Tensor，并移动到设备上
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)  # 转换测试集标签为Tensor，并移动到设备上
    test_loss, test_acc, predicted_emotions = validate(X_test_tensor, y_test_tensor)  # 执行测试
    print(f'Test accuracy is {test_acc:.2f}%')  # 打印测试准确率
