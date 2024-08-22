import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_mic_recorder import mic_recorder
from audio_handler import transcribe_audio

def main():
    st.title("S2T test")
    voice_rec_col = st.sidebar.columns(1)
    # with voice_rec_col:
    voice_recording = mic_recorder(start_prompt="Record Audio", stop_prompt="Stop recording", just_once=True)

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        st.write(transcribed_audio)



if __name__ == "__main__":
    main()