'''
根据语⾳分帧的思想，编写分帧函数。函数定义如下：
函数格式：frameout=enframe(x,win,inc)
输⼊参数：x 是语⾳信号；win 是窗函数，若为窗函数，帧⻓便取窗函数⻓；inc是帧移。
输出参数：frameout是分帧后的数组，⻓度为帧⻓和帧数的乘积
'''

import numpy as np
def enframe(x, win, inc):
    """
    将语⾳信号分帧，并加窗，返回分帧后的数组
    :param x: 语⾳信号
    :param win: 窗函数
    :param inc: 帧移
    :return: 分帧后的数组，⼤⼩为(nframes, len(win))
    """
    x_len = len(x) # 语⾳信号的⻓度
    win_len = len(win) # 窗函数的⻓度：在每⼀帧中，我们需要选择 win_len 个样本点，并对它们进⾏加窗处理
    # 根据窗函数⻓度和帧移计算帧数
    frame_num = int(np.ceil((x_len - win_len + inc) / inc)) # 计算分帧后的帧数：np.ceil()对数组进⾏向上取整【分帧后的帧数可能不是整数】
    pads = ((frame_num - 1) * inc + win_len - x_len) # 计算需要填充的⻓度，保证最后⼀帧⻓度为win的⻓度（填充信号⻓度，使其刚好可以被inc整除）
    if pads > 0: # 如果需要填充，则在信号两端进⾏对称填充
        x = np.pad(x, (0, pads), 'constant')

    # 分帧
    frameout = np.zeros((frame_num, win_len)) # 定义⼀个全0数组，⽤于存储分帧后的结果
    for i in range(frame_num): # 依次对原始语⾳信号进⾏分帧
        frameout[i] = x[i * inc:i * inc + win_len] * win # 计算第i帧的结果，并乘以窗函数win
        
    return frameout # 返回分帧后的结果数组