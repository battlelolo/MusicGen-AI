import os
import torch
from audiocraft.data.audio import audio_write

print("현재 작업 디렉토리:", os.getcwd())

# 무음 오디오 생성
dummy_audio = torch.zeros(1, 16000)
audio_write("test_silence", dummy_audio, 16000)

# 생성 확인
full_path = os.path.abspath("test_silence.wav")
print("기대 저장 경로:", full_path)
print("파일 존재함?", os.path.isfile(full_path))
