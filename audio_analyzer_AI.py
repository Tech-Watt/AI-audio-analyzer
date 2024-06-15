from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import os

# Load environment variables
load_dotenv()
MODEL = 'gpt-4'
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

st.title('AI AUDIO ANALYZER')

# Upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if audio_file:
    # Display audio player
    st.audio(audio_file)

    # Transcribe audio
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )

    # Analyze transcription
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": """You are an audio analyzer AI. 
             Analyze the audio and create a summary of the provided transcription. Respond in Markdown."""},
            {"role": "user", "content": f"The audio transcription is: {transcription.text}"}
        ],
        temperature=0,
    )

    # Display the response
    st.markdown(response.choices[0].message.content)
