import torch
from audiocraft.models import MusicGen
import os
import torchaudio
from datetime import datetime
import re

def sanitize_filename(text):
    # 파일 이름으로 안전하게 변환: 소문자, 공백 → _, 특수문자 제거
    return re.sub(r'[^a-zA-Z0-9_]', '', text.replace(" ", "_").lower())

def custom_audio_write_wav(path: str, wav: torch.Tensor, sample_rate: int):
    now = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_path = f"{now}_{path}.wav"

    torchaudio.save(output_path, wav, sample_rate)
    print("✔️ WAV 저장 완료:", output_path)

# ========== 설정 ==========

duration_sec = 30
prompt = ["deep minimal techno with smooth basslines and clean analog synths, 126 BPM"]
prompt_for_filename = sanitize_filename(prompt[0])[:50]  # 길이 제한 (옵션)

filename_suffix = f"techno_{duration_sec}s_{prompt_for_filename}"

# MusicGen 실행
model = MusicGen.get_pretrained("medium")
model.set_generation_params(duration=duration_sec)

print("🎵 생성 중...")
wav = model.generate(prompt)

# WAV 저장
custom_audio_write_wav(filename_suffix, wav[0].cpu(), model.sample_rate)
