#!/usr/bin/env python3
"""TTS子进程，通过标准输入接收文本，合成语音并播放"""
import sys, os, subprocess

TTS_DIR = '/usr/share/asr-llm-tts'
VENV_SITE = os.path.join(TTS_DIR, '.voice2code', 'lib', 'python3.12', 'site-packages')
sys.path.insert(0, TTS_DIR)
sys.path.insert(0, VENV_SITE)
os.environ['LD_LIBRARY_PATH'] = os.path.join(VENV_SITE, 'onnxruntime', 'capi') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
from tts.melotts.melotts_onnx import TTSModel

SPEAKER = "plughw:2,0"
model = TTSModel(enc_model='encoder-zh.onnx', dec_model='decoder-zh.dynq.onnx')
model.ort_predict("预热")
print("TTS_READY", flush=True)

while True:
    line = sys.stdin.readline()
    if not line: break
    text = line.strip()
    if text == "__EXIT__": break
    if not text: continue
    wav = model.ort_predict(text)
    subprocess.run(["aplay", "-D", SPEAKER, "-q", wav])
    os.unlink(wav)
    print("PLAYBACK_DONE", flush=True)