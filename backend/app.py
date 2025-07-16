import os
import json
import asyncio
import numpy as np
import sounddevice as sd
from queue import Queue
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import torch
from scipy.signal import savgol_filter, find_peaks
import librosa
from collections import deque
import time
import tempfile
import wave
import threading
import requests
from pathlib import Path

# ✅ Load the rich QuickResponse prompt ONCE at startup
quick_response_path = Path(__file__).parent / "QuickResponse.txt"
with open(quick_response_path, "r", encoding="utf-8") as f:
    quick_prompt_template = f.read()

# Single Flask app/CORS/socketio instantiation
app = Flask(__name__, static_folder='../dist')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Removed async_mode

last_agent_transcript = {'transcript': ''}
last_customer_transcript = {'transcript': ''}

# Define this route after app is created
@app.route('/api/conversation', methods=['GET'])
def get_conversation():
    # In a real app, you would fetch these from a database or session
    # Here, we use the last known transcript for each (if available)
    agent_chat = last_agent_transcript.get('transcript', '')
    customer_chat = last_customer_transcript.get('transcript', '')
    return jsonify({
        'agent_chat': agent_chat,
        'customer_chat': customer_chat
    })

from pending_tickets_api import pending_tickets_api
app.register_blueprint(pending_tickets_api)

model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, _, read_audio, _, _) = utils

sample_rate = 16000
frame_duration = 0.5
frame_samples = int(sample_rate * frame_duration)
audio_buffer = Queue()

window_size = 20
pitch_window = deque([0.001]*5, maxlen=window_size)
energy_window = deque([0.001]*5, maxlen=window_size)
rate_window = deque([0.001]*5, maxlen=window_size)

global_stream = None
running = False
x_counter = 0  # ✅ Fixed: Initialize x_counter

# Whisper + Emotion setup
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# --- GPU/CPU selection ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
force_cpu = False  # New flag to force CPU after OOM
print(f"[INFO] Using device: {device}")

# Ensure all models and tensors use the selected device
def to_device(tensor):
    global device
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    return tensor

def load_whisper_model():
    global stt_model, device
    stt_model = whisper.load_model("base", device=device)

def load_emotion_model():
    global emotion_model, device
    emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions").to(device)
    emotion_model.eval()

load_whisper_model()
load_emotion_model()

whisper_emotion = {"emotion": "neutral", "prob": 0.0}
whisper_running = False
whisper_thread = None

BUFFER = Queue()
SAMPLE_RATE = 16000
BLOCK_SECONDS = 2

emotion_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

def recognize_emotion(text):
    global emotion_model, device, force_cpu
    if not text.strip():
        return "neutral", 0.0
    try:
        inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        if force_cpu:
            device = 'cpu'
            emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions").to(device)
            emotion_model.eval()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = emotion_model(**inputs).logits
            probs = F.softmax(logits, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)
        label = emotion_model.config.id2label[top_idx.item()]
        if top_prob.item() < 0.6:
            return "neutral", top_prob.item()
        return label, top_prob.item()
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print('⚠️ CUDA OOM in emotion model, switching to CPU...')
            torch.cuda.empty_cache()
            device = 'cpu'
            force_cpu = True
            load_emotion_model()
            return recognize_emotion(text)
        print(f"⚠️ Emotion detection error: {e}")
        return "error", 0.0


def whisper_audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    BUFFER.put(indata.copy())


def whisper_emotion_loop():
    global whisper_emotion, whisper_running, stt_model, device, force_cpu
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=whisper_audio_callback):
        while whisper_running:
            audio_frames = []
            start_time = time.time()
            while time.time() - start_time < 2 and whisper_running:
                try:
                    chunk = BUFFER.get(timeout=2)
                    audio_frames.append(chunk)
                except Exception:
                    break
            if not audio_frames or not whisper_running:
                continue
            audio_data = np.concatenate(audio_frames)
            # Detect silence/mute: skip if energy is very low
            energy = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            if energy < 100:  # Threshold may need tuning for your mic
                print('[Whisper] Silence/mute detected, skipping transcription.')
                whisper_emotion = {"emotion": "neutral", "prob": 0.0, "transcript": ""}
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                with wave.open(tmpfile, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data.tobytes())
                temp_path = tmpfile.name
            try:
                if force_cpu:
                    device = 'cpu'
                    load_whisper_model()
                result = stt_model.transcribe(temp_path, language="en")
                transcript = result["text"].strip()
                if transcript:
                    emotion, prob = recognize_emotion(transcript)
                    whisper_emotion = {"emotion": emotion, "prob": prob, "transcript": transcript}
                    print(f"[Whisper] Transcript: {transcript}\n[Whisper] Emotion: {emotion} ({prob:.2f})")
            except RuntimeError as e:
                if 'CUDA' in str(e):
                    print('⚠️ CUDA error in Whisper, switching to CPU...')
                    torch.cuda.empty_cache()
                    device = 'cpu'
                    force_cpu = True
                    load_whisper_model()
                    try:
                        result = stt_model.transcribe(temp_path, language="en")
                        transcript = result["text"].strip()
                        if transcript:
                            emotion, prob = recognize_emotion(transcript)
                            whisper_emotion = {"emotion": emotion, "prob": prob, "transcript": transcript}
                            print(f"[Whisper] Transcript: {transcript}\n[Whisper] Emotion: {emotion} ({prob:.2f})")
                    except Exception as e2:
                        print("❌ Whisper/Emotion error after CPU fallback:", e2)
                else:
                    print("❌ Whisper/Emotion error:", e)
            except Exception as e:
                print("❌ Whisper/Emotion error:", e)
            finally:
                os.remove(temp_path)


def start_whisper_emotion():
    global whisper_running, whisper_thread
    if not whisper_running:
        whisper_running = True
        whisper_thread = threading.Thread(target=whisper_emotion_loop, daemon=True)
        whisper_thread.start()
        print("[Whisper] Emotion analysis started.")


def stop_whisper_emotion():
    global whisper_running
    whisper_running = False
    print("[Whisper] Emotion analysis stopped.")


def compute_pitch(audio):
    return float(np.mean(np.abs(np.fft.rfft(audio))))

def compute_energy(audio):
    return float(np.sqrt(np.mean(audio**2)))

def compute_speaking_rate(audio):
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    energy = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
    threshold = np.percentile(energy, 75) * 0.8
    peaks, _ = find_peaks(energy, height=threshold, distance=3)
    syllables = len(peaks)
    wpm = (syllables / 4) / (frame_duration / 60)
    return float(wpm)

def normalize(val, min_val, max_val):
    return min(max(0, ((val - min_val) / (max_val - min_val) * 100) if max_val > min_val else 0), 100)

def get_latest_whisper_emotion():
    # Return both emotion, prob, and transcript if available
    return whisper_emotion.get("emotion", "neutral"), whisper_emotion.get("prob", 0.0), whisper_emotion.get("transcript", "")

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/status')
def status():
    return jsonify({"status": "running"})

@socketio.on('connect')
def on_connect():
    global pitch_window, energy_window, rate_window, x_counter, audio_buffer
    print('Client connected')
    pitch_window.clear()
    energy_window.clear()
    rate_window.clear()
    pitch_window.extend([0.001] * 5)
    energy_window.extend([0.001] * 5)
    rate_window.extend([0.001] * 5)
    while not audio_buffer.empty():
        audio_buffer.get()
    x_counter = 0  # <-- ensure x_counter is reset on connect
    emit('connected', {'data': 'Connected and reset'})

@socketio.on('start_recording')
def on_start_recording():
    print('Starting voice analysis')
    start_voice_analysis()
    start_whisper_emotion()

@socketio.on('stop_recording')
def on_stop_recording():
    print('Stopping voice analysis')
    stop_voice_analysis()
    stop_whisper_emotion()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_buffer.put(indata[:, 0].copy())

def process_audio_graphs():
    global running, x_counter
    print("Started audio processing thread (graphs)")
    while running:
        try:
            if audio_buffer.empty():
                socketio.sleep(0.1)
                continue
            audio_chunk = audio_buffer.get()
            if len(audio_chunk) < frame_samples:
                continue
            audio_np = audio_chunk[:frame_samples]
            torch_audio = torch.from_numpy(audio_np).float()
            if torch.max(torch.abs(torch_audio)) > 1:
                torch_audio = torch_audio / torch.max(torch.abs(torch_audio))
            is_speech = len(get_speech_timestamps(torch_audio, model_vad, sampling_rate=sample_rate)) > 0
            if is_speech:
                pitch = compute_pitch(audio_np)
                energy = compute_energy(audio_np)
                speaking_rate = compute_speaking_rate(audio_np)
                pitch = np.log1p(pitch) * 40
                pitch = min(pitch, 100)
            else:
                pitch = energy = speaking_rate = 0.001
            pitch_window.append(pitch)
            energy_window.append(energy)
            rate_window.append(speaking_rate)
            p_score = normalize(pitch, min(pitch_window), max(pitch_window))
            e_score = normalize(energy, min(energy_window), max(energy_window))
            r_score = normalize(speaking_rate, min(rate_window), max(rate_window))
            data = {
                'time': str(x_counter),
                'pitch': {'value': p_score, 'raw': pitch},
                'energy': {'value': e_score, 'raw': energy},
                'speakingRate': {'value': r_score, 'raw': speaking_rate},
                'emotion': None,
                'is_speech': is_speech
            }
            socketio.emit('graph_data', data)
            x_counter += 1
            socketio.sleep(0.1)
        except Exception as e:
            print(f"Error in graph processing: {e}")
            socketio.sleep(0.5)

def update_quick_response(transcript: str, emotion: str, prob: float = None):
    # (Old docstring prompt removed as it is no longer used)

    # Only skip if transcript is empty, too short, or emotion is neutral with high confidence
    if not transcript or len(transcript.strip()) < 5 or (emotion == "neutral" and (prob is None or prob > 0.95)):
        print(f"[QuickResponse] Skipped: emotion={emotion}, prob={prob}, transcript='{transcript}'")
        return

    try:
        optimized_prompt = (
            "You are assisting a live support agent. Based on the latest customer message in the transcript, generate exactly 3 different short responses the agent could say next. "
            "Each response must be polite, helpful, and straight to the point—no preamble, no explanations, no disclaimers, no repetition. "
            "Each response must be a single line, no more than 10 words, and should guide the conversation forward with clarity. "
            "DO NOT mention that you are an AI, or refer to this prompt or yourself in any way.\n\n"
            "Respond in this format:\n"
            "Response 1: ...\n"
            "Response 2: ...\n"
            "Response 3: ...\n\n"
            f"Transcript:\n{transcript.strip()}"
        )
        print("[QuickResponse] Generating 3 agent responses...")
        print(f"[QuickResponse] Prompt sent to Gemini:\n{optimized_prompt}")
        suggestions = call_ollama_streaming(optimized_prompt)
        print(f"[QuickResponse] Raw Gemini output:\n{suggestions}")
        responses = []
        if suggestions:
            # Try to extract explicit Response 1/2/3 lines
            for line in suggestions.split("\n"):
                orig_line = line
                line = line.strip()
                print(f"[QuickResponse] Checking line: '{orig_line}' -> '{line}'")
                if line.lower().startswith("response 1:") or line.lower().startswith("response1:"):
                    resp = line.split(":", 1)[1].strip()
                    responses.append(resp)
                elif line.lower().startswith("response 2:") or line.lower().startswith("response2:"):
                    resp = line.split(":", 1)[1].strip()
                    responses.append(resp)
                elif line.lower().startswith("response 3:") or line.lower().startswith("response3:"):
                    resp = line.split(":", 1)[1].strip()
                    responses.append(resp)
            # If no explicit responses, fallback: split into sentences and pick up to 3
            if not responses:
                import re
                # Split into sentences (naive, but works for most English)
                sentences = re.split(r'(?<=[.!?])\s+', suggestions.strip())
                # Filter out empty and long sentences
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    words = s.split()
                    if len(words) > 15:
                        continue
                    responses.append(s)
                    if len(responses) == 3:
                        break
        # Post-process: enforce 1 line, at most 10 words, filter meta/preamble
        meta_phrases = [
            "ai assistant", "based on", "responding to", "here's", "here is", "as an ai", "as your ai", "as a call center agent", "output only", "generate", "instruction", "preamble", "meta-response", "sure!", "if you do not understand"
        ]
        filtered = []
        for resp in responses:
            resp_line = resp.split("\n")[0].strip()
            words = resp_line.split()
            if len(words) > 10:
                resp_line = " ".join(words[:10])
            lowered = resp_line.lower()
            if any(phrase in lowered for phrase in meta_phrases) or resp_line.strip().startswith("[a single line") or not resp_line:
                print(f"[QuickResponse] Filtered out meta/preamble or prompt echo: {resp_line}")
                continue
            filtered.append(resp_line)
        if not suggestions:
            print("[QuickResponse] No output from Ollama. Emitting placeholder.")
            socketio.emit('quick_response', {"suggestions": ["(No suggestions from Ollama)"], "suggestion": "(No suggestions from Ollama)"})
        elif not filtered:
            print("[QuickResponse] No direct answers found. Sending fallback message.")
            socketio.emit('quick_response', {"suggestions": ["Sorry, I can't help with that."], "suggestion": "Sorry, I can't help with that."})
        else:
            print("[QuickResponse] Suggestions accepted:")
            for idx, s in enumerate(filtered, 1):
                print(f"  {s}")
            socketio.emit('quick_response', {"suggestions": filtered, "suggestion": "\n".join(filtered)})
    except Exception as e:
        print(f"[QuickResponse] ❌ Error: {e}")
        socketio.emit('quick_response', {"suggestions": [], "suggestion": ""})

def process_audio_emotion():
    global running
    print("Started audio processing thread (emotion)")
    last_transcript = None
    last_emotion = None
    transcript_buffer = []  # Buffer for last 3-5 transcript lines
    last_ollama_time = time.time() - 5  # So first call happens immediately
    while running:
        try:
            whisper_emotion_val, whisper_prob, whisper_transcript = get_latest_whisper_emotion()
            data = {
                'emotion': whisper_emotion_val,
                'prob': whisper_prob,
                'transcript': whisper_transcript,
                'speaker': 'agent'  # Always set speaker for agent
            }
            socketio.emit('emotion_data', data)
            # Buffer management: keep last 5 non-empty lines
            if whisper_transcript and whisper_transcript.strip():
                if not transcript_buffer or whisper_transcript != transcript_buffer[-1]:
                    transcript_buffer.append(whisper_transcript)
                    if len(transcript_buffer) > 5:
                        transcript_buffer.pop(0)
                # Update the agent transcript for the conversation API
                last_agent_transcript['transcript'] = "\n".join(transcript_buffer)
            # Placeholder: If you have customer STT, update last_customer_transcript['transcript'] here
            # last_customer_transcript['transcript'] = ...
            # Batch Ollama call every 5 seconds with latest context
            now = time.time()
            if now - last_ollama_time >= 5:
                context_transcript = "\n".join(transcript_buffer)
                update_quick_response(context_transcript, whisper_emotion_val, whisper_prob)
                last_ollama_time = now
            socketio.sleep(0.2)
        except Exception as e:
            print(f"Error in emotion processing: {e}")
            socketio.sleep(0.5)

def start_voice_analysis():
    global global_stream, running
    running = True
    try:
        global_stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=frame_samples)
        global_stream.start()
        print("Audio stream started successfully")
        socketio.start_background_task(process_audio_graphs)
        socketio.start_background_task(process_audio_emotion)
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        running = False

def stop_voice_analysis():
    global global_stream, running
    running = False
    if global_stream is not None:
        try:
            global_stream.stop()
            global_stream.close()
            print("Audio stream stopped successfully")
        except Exception as e:
            print(f"Error stopping audio stream: {e}")
        global_stream = None

def call_ollama_streaming(prompt, model="llama2", retries=3, delay=5):
    # Use Gemini Pro API for quick responses
    import time
    import requests
    api_key = "AIzaSyBqlhdIKXQfTBLjA8aXSd_whfm808W084o"
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=" + api_key
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=90)
            if response.status_code != 200:
                print(f"[Gemini] Non-200 status: {response.status_code}")
                print(response.text)
                time.sleep(delay)
                continue
            result = response.json()
            # Gemini returns candidates[0].content.parts[0].text
            candidates = result.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()
            print("[Gemini] No valid response structure.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[Gemini] Attempt {attempt + 1}/{retries} failed:", e)
            time.sleep(delay)
    return None



#gowtham code for Issue Complexity Prediction
def call_gemini_api(prompt, retries=3, delay=5):
    import requests
    import time
    api_key = "AIzaSyDvB_7emF_wnTK974irkkaJiuwFv83fKyM" 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=90)
            if response.status_code != 200:
                print(f"[Gemini] Non-200 status: {response.status_code}")
                print(response.text)
                time.sleep(delay)
                continue
            result = response.json()
            candidates = result.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()
            print("[Gemini] No valid response structure.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[Gemini] Attempt {attempt + 1}/{retries} failed:", e)
            time.sleep(delay)
    return None


@app.route('/api/predict_severity', methods=['POST'])
def predict_severity():
    data = request.get_json()
    conversation = data.get('conversation', '')
    if not conversation or len(conversation.strip()) < 5:
        return jsonify({"error": "No valid conversation provided."}), 400

    severity = get_issue_severity(conversation)
    return jsonify({"severity": severity})


def get_issue_severity(conversation):
    prompt = (
        "You are an expert support assistant. "
        "Analyze the following conversation between an agent and an employee. "
        "Based on the conversation only, classify the severity of the employee's issue as 'low', 'medium', or 'high'. "
        "Respond with only one word: low, medium, or high.\n\n"
        f"Conversation:\n{conversation.strip()}"
    )
    response = call_gemini_api(prompt)
    if response:
        for word in ['low', 'medium', 'high']:
            if word in response.lower():
                return word
    return "unknown"



if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)