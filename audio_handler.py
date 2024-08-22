import torch
from transformers import pipeline
from transformers import WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
import librosa
import io
from utils import load_config
config = load_config()

def convert_bytes_to_array(audio_bytes):
    print(audio_bytes)
    print(type(audio_bytes))
    audio_bytes = io.BytesIO(audio_bytes)
    audio, sample_rate = librosa.load(audio_bytes)
    print(sample_rate)
    return audio

def transcribe_audio(audio_bytes):
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", task="transcribe")
    # now switch the prefix token from Spanish to French
    # tokenizer.set_prefix_tokens(language="english")

    forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="english", task="transcribe")
    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-large-v3",
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        chunk_length_s=30,
        device=device,
    )

    audio_array = convert_bytes_to_array(audio_bytes)
    prediction = pipe(audio_array, batch_size=1, generate_kwargs={"forced_decoder_ids": forced_decoder_ids})["text"]

    return prediction


