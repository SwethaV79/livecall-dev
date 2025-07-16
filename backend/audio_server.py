import asyncio
import websockets
import wave
import numpy as np
import tempfile
import torch
import whisper
import os
import scipy.signal
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ----- Model Setup -----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

stt_model = whisper.load_model("base", device=device)
emotion_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions").to(device)
emotion_model.eval()

def recognize_emotion(text):
    global device, emotion_model  # <-- Move here, first line inside the function
    if not text.strip():
        return "neutral", 0.0
    try:
        inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
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
            device = 'cpu'
            emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions").to(device)
            return recognize_emotion(text)
        print(f"⚠️ Emotion detection error: {e}")
        return "error", 0.0

# ----- WebSocket Server -----
connected_clients = set()

CHUNK_DURATION_SEC = 10
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit audio
CHUNK_SIZE = SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_DURATION_SEC  # = 320000 bytes

async def handle_audio(websocket):
    print("Client connected")
    connected_clients.add(websocket)
    buffer = bytearray()
    try:
        async for message in websocket:
            buffer.extend(message)
            while len(buffer) >= CHUNK_SIZE:
                # Convert bytes to numpy array (int16)
                audio = np.frombuffer(buffer[:CHUNK_SIZE], dtype=np.int16)
                orig_sr = 48000  # Adjust if your client uses a different sample rate
                # Resample to 16000
                audio_resampled = scipy.signal.resample_poly(audio, 16000, orig_sr)
                audio_resampled = audio_resampled.astype(np.int16)
                # Write to temp wav
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                    with wave.open(temp_wav.name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(BYTES_PER_SAMPLE)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(audio_resampled.tobytes())
                transcript = await transcribe_audio(temp_wav.name)
                if transcript.strip() and transcript != "(silence)":
                    emotion, prob = recognize_emotion(transcript)
                    result = {
                        "emotion": emotion,
                        "prob": round(float(prob), 2),
                        "transcript": transcript,
                        "speaker": "employee"
                    }
                    await broadcast_output(result)
                else:
                    print("[INFO] Skipped silence or empty transcript.")
                os.remove(temp_wav.name)
                buffer = buffer[CHUNK_SIZE:]
    finally:
        connected_clients.remove(websocket)

async def transcribe_audio(path):
    try:
        result = stt_model.transcribe(path, language="en")
        return result["text"].strip() or "(silence)"
    except Exception as e:
        return f"[Error] {e}"

async def broadcast_output(obj):
    msg = json.dumps(obj, ensure_ascii=False)
    for client in connected_clients.copy():
        try:
            await client.send(msg)
        except:
            pass

async def main():
    async with websockets.serve(handle_audio, "localhost", 7000):
        print("✅ Employee WebSocket server running at ws://localhost:7000")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())