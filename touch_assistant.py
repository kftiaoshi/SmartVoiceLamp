#!/root/lianwang/bin/python3
# -*- coding: utf-8 -*-
"""小宝语音助手 - 主程序"""
import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog, messagebox
import subprocess, threading, re, requests, time, os, sys, json, numpy as np
from datetime import datetime, timedelta
import lgpio
import scipy.signal
import pickle
from sherpa_onnx import SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig
from smbus2 import SMBus

# ---------- 软件PWM LED控制 ----------
class SoftwarePWM:
    def __init__(self, gpio=71):
        self.gpio = gpio
        self._brightness = 0
        self._running = True
        self._lock = threading.Lock()
        self._callback = None
        self._handle = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._handle, gpio)
        lgpio.gpio_write(self._handle, gpio, 0)
        self._thread = threading.Thread(target=self._pwm_loop, daemon=True)
        self._thread.start()

    def _pwm_loop(self):
        period = 1.0 / 200
        while self._running:
            with self._lock:
                b = self._brightness
            if b <= 0:
                lgpio.gpio_write(self._handle, self.gpio, 0)
                time.sleep(0.1)
            elif b >= 100:
                lgpio.gpio_write(self._handle, self.gpio, 1)
                time.sleep(0.1)
            else:
                hi = period * b / 100
                lo = period - hi
                lgpio.gpio_write(self._handle, self.gpio, 1)
                time.sleep(hi)
                lgpio.gpio_write(self._handle, self.gpio, 0)
                time.sleep(lo)

    def set(self, v):
        self._brightness = max(0, min(100, v))
        if self._callback:
            self._callback(self._brightness)

    def get(self):
        return self._brightness

    def set_callback(self, callback):
        self._callback = callback

    def stop(self):
        self._running = False
        self._thread.join(timeout=1)
        lgpio.gpio_write(self._handle, self.gpio, 0)
        lgpio.gpiochip_close(self._handle)

led = SoftwarePWM(71)

# ---------- BH1750 光照传感器 ----------
class BH1750:
    def __init__(self, bus=4, addr=0x23):
        self.bus = SMBus(bus)          # 根据实际I2C总线修改bus号
        self.addr = addr
        self._init()

    def _init(self):
        self.bus.write_byte(self.addr, 0x01)   # 上电
        time.sleep(0.01)
        self.bus.write_byte(self.addr, 0x07)   # 复位
        time.sleep(0.01)

    def read_lux(self):
        self.bus.write_byte(self.addr, 0x20)   # 单次高分辨率测量模式
        time.sleep(0.2)
        data = self.bus.read_i2c_block_data(self.addr, 0x00, 2)
        raw = (data[0] << 8) | data[1]
        return round(raw / 1.2, 1)

    def close(self):
        self.bus.close()

bh1750 = None
try:
    bh1750 = BH1750(bus=4, addr=0x23)  # 实际I2C总线根据硬件连接调整
    print("[系统] BH1750 光照传感器已连接")
except Exception as e:
    print(f"[系统] BH1750 未连接: {e}")

# ---------- 声纹管理器 ----------
class VPManager:
    def __init__(self, model_path, db_path="/root/speakers_v2.pkl"):
        config = SpeakerEmbeddingExtractorConfig()
        config.model = model_path
        self.extractor = SpeakerEmbeddingExtractor(config)
        self.db_path = db_path
        self._speakers = {}
        self._audio_map = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "speakers" in data:
                    self._speakers = data["speakers"]
                    self._audio_map = data.get("audio", {})
                else:
                    self._speakers = data
                    self._audio_map = {}
            except:
                self._speakers = {}
                self._audio_map = {}

    def _save(self):
        try:
            data = {"speakers": self._speakers, "audio": self._audio_map}
            with open(self.db_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"[声纹] 保存失败: {e}")

    def register(self, name, wav_path):
        emb = self._extract(wav_path)
        if emb is None:
            return False, "特征提取失败"
        with self._lock:
            self._speakers.setdefault(name, []).append(emb)
            self._audio_map.setdefault(name, []).append(wav_path)
            self._save()
        return True, f"{name} 已注册"

    def verify(self, wav_path, threshold=0.75):
        emb = self._extract(wav_path)
        if emb is None:
            return {"ok": False, "score": 0, "msg": "特征提取失败"}
        with self._lock:
            if not self._speakers:
                return {"ok": False, "score": 0, "msg": "声纹库为空"}
            best_score = 0.0
            best_name = None
            for name, emb_list in self._speakers.items():
                for stored_emb in emb_list:
                    dot = np.dot(emb, stored_emb)
                    norm = np.linalg.norm(emb) * np.linalg.norm(stored_emb)
                    if norm == 0:
                        continue
                    score = dot / norm
                    if score > best_score:
                        best_score = score
                        best_name = name
        pct = round(best_score * 100, 2)
        if best_score >= threshold:
            return {"ok": True, "score": pct, "msg": f"匹配: {best_name}"}
        return {"ok": False, "score": pct, "msg": "无匹配说话人"}

    def list_users(self):
        with self._lock:
            result = []
            for name in self._speakers:
                audio_list = self._audio_map.get(name, [])
                audio_path = audio_list[0] if audio_list else ""
                result.append({"name": name, "time": "", "audio": audio_path})
            return result

    def delete(self, name):
        with self._lock:
            if name in self._speakers:
                del self._speakers[name]
                self._audio_map.pop(name, None)
                self._save()
                return True
        return False

    def _extract(self, wav_path):
        try:
            import soundfile as sf
            audio, rate = sf.read(wav_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            arr = audio.astype(np.float32)
            max_val = np.max(np.abs(arr))
            if max_val > 0:
                arr = arr / max_val * 0.9
            if rate == 48000:
                n = int(len(arr) * 16000 / 48000)
                arr = scipy.signal.resample(arr, n).astype(np.float32)
            stream = self.extractor.create_stream()
            stream.accept_waveform(16000, arr)
            stream.input_finished()
            return np.array(self.extractor.compute(stream))
        except Exception as e:
            print(f"[声纹] 提取失败: {e}")
            return None

vp_mgr = VPManager(model_path="/root/sherpa-models/speaker/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")

def call_vp(*args):
    cmd = args[0]
    if cmd == "register":
        name = args[1]
        for f in args[2:]:
            vp_mgr.register(name, f)
        return {"ok": True, "msg": f"{name} 注册成功"}
    elif cmd == "verify":
        return vp_mgr.verify(args[1], threshold=0.75)
    elif cmd == "list":
        return vp_mgr.list_users()
    elif cmd == "delete":
        vp_mgr.delete(args[1])
        return {"ok": True, "msg": f"{args[1]} 已删除"}
    return {"ok": False, "msg": "未知命令"}

# ---------- 配置 ----------
SPEAKER_DEVICE = "plughw:2,0"          # USB扬声器设备
MIC_DEVICE_INDEX = 2                   # USB麦克风设备索引
WAKEUP_TIMEOUT = 20                     # 唤醒后无操作自动退出时间（秒）

# DeepSeek API 配置
AI_API_URL = "https://api.deepseek.com/v1/chat/completions"
AI_API_KEY = "YOUR_DEEPSEEK_API_KEY"   # 替换为你的DeepSeek API密钥

wakeup_pattern = re.compile(r'(小\s*[宝吧八把巴])')

# ASR误识别纠错字典
asr_corrections = {
    '这是第意思啊': '现在几点', '现在是多少点': '现在几点',
    '小狗说话': '小宝说话', '我心你': '我问你', '醒我': '提醒我',
    '提醒我修': '提醒我休息', '提醒我立经出': '提醒我立即喝水',
    '一分钟后提醒我修': '一分钟后提醒我休息', '醒我一分钟后吃药': '提醒我一分钟后吃药',
}

# 城市名称到拼音的映射，用于天气查询
CITY_MAP = {
    "北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou", "深圳": "Shenzhen",
    "成都": "Chengdu", "重庆": "Chongqing", "杭州": "Hangzhou", "武汉": "Wuhan",
    "西安": "Xian", "南京": "Nanjing", "天津": "Tianjin", "长沙": "Changsha",
    "郑州": "Zhengzhou", "济南": "Jinan", "哈尔滨": "Harbin", "沈阳": "Shenyang",
    "昆明": "Kunming", "贵阳": "Guiyang", "苏州": "Suzhou", "厦门": "Xiamen",
    "青岛": "Qingdao", "大连": "Dalian", "宁波": "Ningbo", "福州": "Fuzhou",
    "合肥": "Hefei", "南宁": "Nanning", "南昌": "Nanchang", "太原": "Taiyuan",
    "石家庄": "Shijiazhuang", "乌鲁木齐": "Urumqi", "兰州": "Lanzhou", "海口": "Haikou",
    "呼和浩特": "Hohhot", "银川": "Yinchuan", "西宁": "Xining", "拉萨": "Lhasa",
    "香港": "Hong Kong", "澳门": "Macau", "台北": "Taipei",
}

# 灯光控制关键词库
LIGHT_ON_KEYWORDS = ['开灯','打开灯','点亮','亮灯','把灯打开','请开灯','开一下灯',
                     '灯光开启','开个灯','让灯亮','亮起来','点亮灯光','照明','开开灯',
                     '灯亮一下','把灯点亮','来点光','给点光','开启灯光','开亮','灯开',
                     '灯打开','亮起','灯亮起']
LIGHT_OFF_KEYWORDS = ['关灯','关闭灯','关掉','熄灭','关掉灯','把灯关了','灯关掉',
                      '灯光关闭','关个灯','灭灯','关一下灯','关灯吧','关灯了',
                      '关灯睡觉','不需要光','太亮了关掉','关灯休息','熄灯',
                      '关闭灯光','关亮','灯关','灯关闭','灭掉','把灯熄灭']
LIGHT_MAX_KEYWORDS = ['最亮','开到最亮','调到最亮','亮度最高','最大亮度','最亮模式',
                      '灯光最亮','把灯调到最亮','开到最大','调到最大','亮度调到最高',
                      '开到最亮','调到最亮','调到最大','开到最大','最亮灯',
                      '最高亮度','亮度调到顶','把亮度调到最高','把灯调到最大',
                      '亮度最大','灯最亮','最亮的灯','调最亮','开最亮',
                      '亮度调到最亮','灯调到最亮','调节到最亮','亮度调至最亮',
                      '将亮度调到最大','把亮度调到最大','把灯亮度调到最大',
                      '亮度开到最大','亮度调到最大','调到一百','开到一百',
                      '亮度一百','百分之百亮','调到顶']
LIGHT_MIN_KEYWORDS = ['最暗','开到最暗','调到最暗','最低亮度','最小亮度',
                      '灯光最暗','把灯调到最暗','开到最小','调到最小',
                      '亮度最小','灯最暗','最暗的灯','调最暗','开最暗',
                      '调到最暗','开到最小','调到最小','最暗模式',
                      '亮度调到最低','灯调到最暗','调节到最暗','亮度调至最暗',
                      '将亮度调到最小','把亮度调到最小','亮度调到最小',
                      '调到零','开到零','亮度零','百分之零亮','调到没',
                      '关到最小','微亮','最暗灯']
LIGHT_BRIGHTER_KEYWORDS = ['亮一点','变亮','调亮','太暗','暗了','亮度调高','把亮度调高',
                           '把灯调亮一点','调亮一点','亮一些','增亮','提高亮度','加点亮度',
                           '调亮点','增加亮度','亮些','亮一度','亮一点儿','再加亮',
                           '提高一点亮度','调高亮度','亮度加','亮度增加','亮一下']
LIGHT_DIMMER_KEYWORDS = ['暗一点','变暗','调暗','太亮','刺眼','亮度调低','把亮度调低',
                         '把灯调暗一点','调暗一点','暗一些','降低亮度','减点亮度',
                         '调暗点','减少亮度','暗些','暗一度','暗一点儿','再暗一点',
                         '降低一点亮度','调低亮度','亮度减','亮度减少','暗一下']

def understand(text):
    # 解析自然语言指令，返回意图和参数
    if any(k in text for k in LIGHT_ON_KEYWORDS):
        return ("light", {"action": "on"})
    if any(k in text for k in LIGHT_OFF_KEYWORDS):
        return ("light", {"action": "off"})
    if any(k in text for k in LIGHT_MAX_KEYWORDS):
        return ("light", {"action": "set", "value": 100})
    if any(k in text for k in LIGHT_MIN_KEYWORDS):
        return ("light", {"action": "set", "value": 0})
    m = re.search(r'(?:亮度|调到|开到|调至|调节到|把亮度调到|调到百分之|开到百分之|亮度百分之|开到|调到|亮度到)\s*([零一二两三四五六七八九十\d]+)\s*(?:%|百分之|度)?', text)
    if not m: m = re.search(r'百分之\s*([零一二两三四五六七八九十\d]+)', text)
    if not m: m = re.search(r'亮度\s*([零一二两三四五六七八九十\d]+)\s*$', text)
    if m:
        try: val = int(m.group(1))
        except: val = chinese_to_int(m.group(1))
        if 0 <= val <= 100: return ("light", {"action": "set", "value": val})
    if any(k in text for k in LIGHT_BRIGHTER_KEYWORDS):
        return ("light", {"action": "brighter"})
    if any(k in text for k in LIGHT_DIMMER_KEYWORDS):
        return ("light", {"action": "dimmer"})
    if any(k in text for k in ['音量调大','增大','大点声','音量调小','减小','小点声']):
        return ("volume", {"text": text})
    if any(k in text for k in ['自动调光', '自动亮度', '光感模式', '自动灯光', '自动模式']):
        return ("auto_light", {"action": "toggle"})
    return ("chat", {"question": text})

def get_human_time():
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"今天是{now.year}年{now.month}月{now.day}日，{weekdays[now.weekday()]}，现在时间{now.strftime('%H:%M')}"

chinese_num_map = {'零':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,'半':0.5}
def chinese_to_int(s):
    s = s.strip()
    if s in chinese_num_map: return chinese_num_map[s]
    if '十' in s:
        parts = s.split('十')
        left = chinese_num_map.get(parts[0], 0) if parts[0] else 1
        right = chinese_num_map.get(parts[1], 0) if parts[1] else 0
        return left * 10 + right
    try: return float(s)
    except: return 0

def is_online():
    try:
        requests.get("https://www.baidu.com", timeout=2)
        return True
    except:
        return False

def call_ai_with_history(messages):
    """调用DeepSeek API进行对话"""
    api_key = AI_API_KEY   # 从配置中读取API密钥
    print("[AI] 请求 DeepSeek...")
    try:
        r = requests.post(AI_API_URL,
                          headers={"Authorization": f"Bearer {api_key}"},
                          json={"model": "deepseek-chat", "messages": messages},
                          timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[AI] 异常: {e}")
    return "小宝正在思考，请稍后再问吧～"

class WifiDialog(tk.Toplevel):
    # WiFi管理对话框（与完整版一致，此处省略具体实现，保留原有代码）
    pass

class App:
    def __init__(self, root):
        # 初始化界面、TTS、传感器、线程等，完整代码请参考原始文件
        pass

    # 其余方法（_auto_light_loop, _speak_text, _listener_loop, _do_register等）保持不变
    # 请将完整实现补充在此处

if __name__ == '__main__':
    for attempt in range(3):
        try:
            root = tk.Tk(); break
        except tk.TclError as e:
            if 'no display' in str(e).lower():
                print(f"[系统] 等待图形界面（第{attempt+1}次）..."); time.sleep(2)
            else: raise
    else: sys.exit(1)
    app = App(root)
    try: root.mainloop()
    finally:
        if app.reminder_timer: app.reminder_timer.cancel()
        if app.reminder_repeat_timer: app.reminder_repeat_timer.cancel()
        if app.tts_proc: app.tts_proc.terminate()
        led.stop()
        if bh1750: bh1750.close()