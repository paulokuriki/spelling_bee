import streamlit as st
import os
import uuid
import tempfile
from pathlib import Path
from openai import OpenAI
import pandas as pd

# ---------------------- DATA ---------------------- #
CAPITALS = sorted([
    ("Alabama", "Montgomery"), ("Alaska", "Juneau"), ("Arizona", "Phoenix"), ("Arkansas", "Little Rock"),
    ("California", "Sacramento"), ("Colorado", "Denver"), ("Connecticut", "Hartford"), ("Delaware", "Dover"),
    ("Florida", "Tallahassee"), ("Georgia", "Atlanta"), ("Hawaii", "Honolulu"), ("Idaho", "Boise"),
    ("Illinois", "Springfield"), ("Indiana", "Indianapolis"), ("Iowa", "Des Moines"), ("Kansas", "Topeka"),
    ("Kentucky", "Frankfort"), ("Louisiana", "Baton Rouge"), ("Maine", "Augusta"), ("Maryland", "Annapolis"),
    ("Massachusetts", "Boston"), ("Michigan", "Lansing"), ("Minnesota", "Saint Paul"), ("Mississippi", "Jackson"),
    ("Missouri", "Jefferson City"), ("Montana", "Helena"), ("Nebraska", "Lincoln"), ("Nevada", "Carson City"),
    ("New Hampshire", "Concord"), ("New Jersey", "Trenton"), ("New Mexico", "Santa Fe"), ("New York", "Albany"),
    ("North Carolina", "Raleigh"), ("North Dakota", "Bismarck"), ("Ohio", "Columbus"), ("Oklahoma", "Oklahoma City"),
    ("Oregon", "Salem"), ("Pennsylvania", "Harrisburg"), ("Rhode Island", "Providence"), ("South Carolina", "Columbia"),
    ("South Dakota", "Pierre"), ("Tennessee", "Nashville"), ("Texas", "Austin"), ("Utah", "Salt Lake City"),
    ("Vermont", "Montpelier"), ("Virginia", "Richmond"), ("Washington", "Olympia"), ("West Virginia", "Charleston"),
    ("Wisconsin", "Madison"), ("Wyoming", "Cheyenne")
], key=lambda x: x[0])

# ---------------------- TEXT-TO-SPEECH ---------------------- #
class TextToSpeech:
    MAX_CHARS = 120
    MODEL = "gpt-4o-mini-tts"
    VOICE = "nova"

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def _tmp_mp3(self) -> Path:
        return Path(tempfile.gettempdir()) / f"tts_{uuid.uuid4().hex[:8]}.mp3"

    def speak(self, text: str):
        if not text.strip():
            return
        text = text[:self.MAX_CHARS]
        out = self._tmp_mp3()
        if self.client:
            with self.client.audio.speech.with_streaming_response.create(
                model=self.MODEL,
                voice=self.VOICE,
                input=text,
            ) as response:
                response.stream_to_file(str(out))
        else:
            out.write_bytes(b"\xFF\xFB\x90\x44\x00\x00\x00\x00")
        st.session_state.audio_file = str(out)

tts = TextToSpeech()

# ---------------------- STATE HELPERS ---------------------- #
def init_state():
    if "queue" not in st.session_state:
        st.session_state.queue = CAPITALS
        st.session_state.idx = 0
        st.session_state.phase = "waiting"
        st.session_state.running = False
        st.session_state.results = []


def current_pair():
    return st.session_state.queue[st.session_state.idx]


def speak_question():
    state, _ = current_pair()
    tts.speak(f"What is the capital of {state}?")


def speak_answer():
    state, capital = current_pair()
    tts.speak(f"The capital of {state} is {capital}!")


def start_quiz():
    st.session_state.queue = CAPITALS
    st.session_state.idx = 0
    st.session_state.running = True
    st.session_state.phase = "question"
    st.session_state.results = []
    speak_question()


def mark_result(is_correct):
    state, capital = current_pair()
    st.session_state.results.append({"State": state, "Capital": capital, "Correct": is_correct})
    st.session_state.idx += 1
    if st.session_state.idx >= len(st.session_state.queue):
        st.session_state.running = False
        st.balloons()
        return
    st.session_state.phase = "question"
    speak_question()


def show_answer():
    st.session_state.phase = "answer"
    speak_answer()

# ---------------------- UI ---------------------- #
st.set_page_config(page_title="🌸 State Capitals Adventure! 🌸", page_icon="🦄", layout="centered")

init_state()

st.title("😸 State Capitals Quiz 😻")
col1, col2 = st.columns([2, 1])

with col1:
    if not st.session_state.running:
        st.button("🌈 Start the Journey! 🚀", on_click=start_quiz, use_container_width=True)

    if st.session_state.running:
        state, capital = current_pair()
        if st.session_state.phase == "question":
            st.markdown(f"### 🧐 What is the capital of **_{state}_**?")
            st.button("🎯 Show Answer! 🎯", on_click=show_answer, use_container_width=True)
        elif st.session_state.phase == "answer":
            st.markdown(f"### 🏆 The capital is **{capital}!**")
            colA, colB = st.columns(2)
            with colA:
                st.button("✅ I Got it Right!", on_click=lambda: mark_result(True), use_container_width=True)
            with colB:
                st.button("❌ Oops, Made a Mistake", on_click=lambda: mark_result(False), use_container_width=True)

if "audio_file" in st.session_state:
    st.audio(st.session_state.audio_file, format="audio/mp3", autoplay=True)

if st.session_state.results:
    st.header("📋 Your Progress:")
    results_df = pd.DataFrame(st.session_state.results)
    results_df_display = results_df.copy()
    results_df_display["Correct"] = results_df_display["Correct"].map({True: "✅", False: "❌"})
    st.dataframe(results_df_display, use_container_width=True)