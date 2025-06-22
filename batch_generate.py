import torch
from audiocraft.models import MusicGen
import torchaudio
from datetime import datetime
import re

def custom_audio_write_wav(path: str, wav: torch.Tensor, sample_rate: int):
    torchaudio.save(path, wav, sample_rate)
    print("✔️ 저장 완료:", path)

# 지직거림 줄인 테크노 프롬프트들
prompts = [
    "warm dub techno with soft reverb, mellow chords, and steady groove",
    "atmospheric Berlin-style techno with clean percussion and lush textures",
    "melodic warehouse techno with spacious pads and punchy but clean kick",
    "hypnotic techno with filtered synth stabs, subtle delays, and no distortion",
    "modular ambient techno, warm tape textures, and deep rhythmic patterns",
    "spacious deep techno with dreamy layers and clean production, 124 BPM",
    "classic Detroit techno with crisp drums, mellow chords, and analog vibe"
]

# 모델 로드 및 설정
model = MusicGen.get_pretrained("medium")
model.set_generation_params(duration=30)

for prompt in prompts:
    print(f"🎵 생성 중: {prompt}")
    wav = model.generate([prompt])[0].cpu()

    # 파일 이름 정제 (특수문자 제거 & 20자 제한)
    short_prompt = re.sub(r'[^a-zA-Z0-9 ]', '', prompt)[:20].replace(" ", "_")
    now = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"{now}_{short_prompt}.wav"

    custom_audio_write_wav(filename, wav, model.sample_rate)
