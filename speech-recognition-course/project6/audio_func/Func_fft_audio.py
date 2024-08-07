import numpy as np
def fft_func(audio,N,fs):
    #noise=1.2+np.sin(440*2*np.pi*time+np.pi/4)+2.5*np.cos(4000*2*np.pi*time+np.pi/4)
    fft_noise = np.fft.fft(audio)
    #实信号fft的结果前半部分对应[0, fs/2]是正频率的结果,后半部分对应[ -fs/2, 0]是负频率的结果，只需选取前N/2+1个点，也可以用np.fft.fftshift()实现
    #fft_noise = np.fft.fftshift(fft_noise)
    A_fft_noise = abs(fft_noise)

    A_adj = np.zeros((N+1)//2)
    A_adj = A_fft_noise[0:(N+1)//2]
    #对复数幅值修正：复数序列的幅值要进行以下转换，才能得到时域中对应信号的幅值
    A_adj[0]=A_adj[0]/N
    A_adj[-1]=A_adj[-1]/N
    A_adj[1:-1]=2*A_adj[1:-1]/N

    # 频域横轴
    Freq = np.arange(0, N) * fs / N

    return Freq,A_adj
