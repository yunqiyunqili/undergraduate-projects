import wave
import subprocess
def play_audio(wave_path):

    chunk = 1024 #声卡设置的帧数
    wf = wave.open(wave_path, 'rb')
    p = subprocess.call(['afplay',wave_path])