import torch
from audiocraft.models import MusicGen
import subprocess
import tempfile
import os
import torchaudio
from datetime import datetime

def custom_audio_write_mp3(path: str, wav: torch.Tensor, sample_rate: int):
    temp_pcm = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_pcm.name
    temp_pcm.close()
    torchaudio.save(temp_path, wav, sample_rate)

    ffmpeg_path = r"C:\Users\leeyl\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
    now = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_path = f"{now}_{path}.mp3"

    result = subprocess.run([
        ffmpeg_path, "-y", "-i", temp_path, "-ar", str(sample_rate),
        "-codec:a", "libmp3lame", "-b:a", "192k", output_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        os.remove(temp_path)
    except Exception as e:
        print(f"⚠️ 임시 파일 삭제 실패: {e}")

    print("✔️ MP3 저장 완료:", output_path)


# ========== 설정 ==========
duration_sec = 30  # 🔁 여기만 바꾸면 됨!
prompt = ["distorted industrial techno, 909 kick and metallic FX, cold Berlin vibe"]
filename_suffix = f"techno_{duration_sec}s"

# MusicGen 실행
model = MusicGen.get_pretrained("medium")
model.set_generation_params(duration=duration_sec)

print("🎵 생성 중...")
wav = model.generate(prompt)

# MP3 저장
custom_audio_write_mp3(filename_suffix, wav[0].cpu(), model.sample_rate)
