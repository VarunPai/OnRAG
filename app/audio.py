import os
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import warnings

# Suppress FP16 warning if running on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def record_audio(duration: int = 5, sample_rate: int = 44100, filename: str = "temp_recording.wav"):
    """Records audio from the microphone."""
    print(f"--- Recording for {duration} seconds... ---")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("--- Recording complete. ---")
    write(filename, sample_rate, recording)
    return filename


def transcribe_audio_locally(file_path: str):
    """Transcribes audio using local Whisper model."""
    print(f"--- Loading local Whisper model... ---")
    
    # Models: 'tiny', 'base', 'small', 'medium', 'large'
    model = whisper.load_model("base")
    
    print(f"--- Transcribing locally... ---")
    result = model.transcribe(file_path, fp16=False) # fp16=False if you don't have a GPU
    
    return result["text"].strip()


if __name__ == "__main__":
    audio_file = record_audio(duration=5)
    
    text = transcribe_audio_locally(audio_file)
    
    print("\n" + "="*30)
    print(f"LOCAL TRANSCRIPTION: {text}")
    print("="*30)

    # Clean up
    if os.path.exists(audio_file):
        os.remove(audio_file)
