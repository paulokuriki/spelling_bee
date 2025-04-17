import streamlit as st
import random
from datetime import datetime
import base64
import os
import re
import tempfile
import uuid
import time
import wave
from pathlib import Path
from time import time
from typing import Optional, Tuple


# Import OpenAI for text-to-speech and speech-to-text
from openai import OpenAI

# Set page configuration
st.set_page_config(
    page_title="Spelling Bee Practice",
    page_icon="🐝",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import hmac

def check_password():
    """Returns `True` if the user entered a correct password."""

    def login_form():
        """Streamlit-only login form with improved UX."""
        _, c2, _ = st.columns([1, 2, 1])
        with c2:
            with st.container(border=False):

                st.write("# 🔐 Login")
                st.write("Please enter your credentials to access the dashboard.")

                # Centering with columns
                #col1, col2, col3 = st.columns([1, 2, 1])
                #with col2:
                # Username and Password Fields inside a form
                with st.form("Login Form"):
                    st.text_input("Username", key="username", placeholder="Enter your username")
                    st.text_input("Password", type="password", key="password", placeholder="Enter your password")
                    submit_button = st.form_submit_button("Log in")

                    # Call the password check if button is clicked
                    if submit_button:
                        password_entered()

    def password_entered():
        """Checks if the entered password is correct."""
        if st.session_state["username"] in st.secrets["passwords"] and hmac.compare_digest(
            st.session_state["password"], st.secrets.passwords[st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username and password are validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username and password.
    login_form()
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 User not known or password incorrect")
    return False

# Call the `check_password` function to display the login page
if not check_password():
    st.stop()  # Stop execution if login is unsuccessful



# Constants
MAX_DURATION_AUDIO = 15  # Maximum duration of audio recording in seconds


# TextToSpeech class using OpenAI
class TextToSpeech:
    """Class for converting text to speech using OpenAI's TTS model."""

    # Constants
    MAX_TTS_CHARS = 200  # setting 200 to avoid long speeches
    DEFAULT_VOICE = "coral"  # OpenAI voice
    DEFAULT_MODEL = "gpt-4o-mini-tts"  # OpenAI model

    def __init__(self):
        """Initialize TTS with OpenAI."""
        # Initialize OpenAI client
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = None

        # Try to initialize OpenAI client
        try:
            if self.api_key:
                self.openai_client = OpenAI(api_key=self.api_key)
                self._log("OpenAI client initialized successfully")
            else:
                self._log("OpenAI API key not found in environment variables")
                self.mock_tts = True
        except Exception as e:
            self._log(f"Error initializing OpenAI client: {str(e)}", error=True)
            self.mock_tts = True

        # Initialize session state for TTS
        if 'enable_tts' not in st.session_state:
            st.session_state.enable_tts = True
        if 'processing' not in st.session_state:
            st.session_state.processing = False

        # For demo purposes when API key isn't available
        self.mock_tts = not self.openai_client

    def _log(self, message: str, error: bool = False) -> None:
        """Log messages to console and optionally show errors in Streamlit."""
        print(message)
        if error and not getattr(self, 'mock_tts', False):  # Don't show errors for mock TTS
            st.error(message)

    def _preprocess_text(self, text: str, max_length: int) -> str:
        """
        Preprocess text for TTS:
        - Truncate to max length
        - Add pauses after punctuation

        Args:
            text: Input text
            max_length: Maximum character limit

        Returns:
            Preprocessed text
        """
        # Truncate if needed
        if len(text) > max_length:
            self._log(f"Text too long ({len(text)} chars), truncating to {max_length} chars")
            text = text[:max_length]

        # Add pauses for better speech comprehension
        text = re.sub(r'([.!?])\s+', r'\1, ', text)

        return text

    def _create_temp_file(self) -> Tuple[str, Path]:
        """
        Create a temporary file for storing audio.

        Returns:
            Tuple of (filename, full path)
        """
        unique_id = str(uuid.uuid4())[:8]
        filename = f"tts_response_{unique_id}.mp3"
        temp_dir = tempfile.gettempdir()
        temp_file = Path(temp_dir) / filename
        self._log(f"Temp file path: {temp_file}")
        return filename, temp_file

    def synthesize_speech_openai(self, text: str, voice: str = None, model: str = None) -> Optional[str]:
        """
        Convert text to speech using OpenAI's TTS.

        Args:
            text: The text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer, etc.)
            model: TTS model to use

        Returns:
            Path to the audio file or None if unsuccessful
        """
        # Use default values if not specified
        voice = voice or self.DEFAULT_VOICE
        model = model or self.DEFAULT_MODEL

        # Check if OpenAI client is available
        if not self.openai_client:
            self._log("OpenAI client not available")
            return None

        try:
            # Create temp file
            _, temp_file = self._create_temp_file()

            # Preprocess text
            text = self._preprocess_text(text, self.MAX_TTS_CHARS)

            # Generate speech
            self._log(f"Calling OpenAI TTS API with text: '{text}'")
            start_time = time()

            # Use streaming response for better performance
            with self.openai_client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=voice,
                    input=text,
                    instructions="Speak naturally and clearly for a spelling bee exercise."
            ) as response:
                response.stream_to_file(str(temp_file))

            self._log(f"OpenAI TTS API call took {time() - start_time:.2f} seconds")
            self._log(f"File size: {os.path.getsize(temp_file)} bytes")

            return str(temp_file)

        except Exception as e:
            self._log(f"Error generating speech with OpenAI: {str(e)}", error=True)
            return None

    def generate_audio(self, text: str) -> Optional[str]:
        """
        Convert text to speech using OpenAI or fallback to mock.

        Args:
            text: The text to convert to speech

        Returns:
            Path to the audio file or None if unsuccessful
        """
        # If we have OpenAI client, use it
        if not self.mock_tts:
            return self.synthesize_speech_openai(text, voice=st.session_state.voice)

        # Otherwise, create a mock audio file for demo purposes
        try:
            # Create a placeholder audio file for demo
            _, temp_file = self._create_temp_file()

            # For demo purposes, let's create a small silent MP3 file
            demo_audio_path = "demo_audio.mp3"
            if os.path.exists(demo_audio_path):
                with open(demo_audio_path, "rb") as src, open(temp_file, "wb") as dst:
                    dst.write(src.read())
            else:
                # If demo file doesn't exist, create minimal MP3 file
                with open(temp_file, "wb") as f:
                    # Minimal MP3 header (not a real MP3, just for demo)
                    f.write(b"\xFF\xFB\x90\x44\x00\x00\x00\x00")

            print(f"Mock audio file created at {temp_file}")
            return str(temp_file)

        except Exception as e:
            self._log(f"Error generating mock speech: {str(e)}", error=True)
            return None

    def convert_tts(self, text: str) -> None:
        """
        Convert text to speech and prepare it for playback in Streamlit.

        Args:
            text: The text to convert to speech
        """
        try:
            # Ensure we're not processing empty text
            if not text.strip():
                print("Warning: Empty text provided for speech synthesis")
                return

            print(f"Processing speech synthesis for text: '{text}'")

            # Format the text as a spelling instruction
            clean_text = f"Spell the word: {text}"

            # Convert text to speech
            audio_file = self.generate_audio(clean_text)

            if audio_file:
                print(f"Audio file created: {audio_file}")
                # Store in session state for the audio player to use
                st.session_state.audio_file = audio_file
            else:
                # Clean up session state if synthesis failed
                if "audio_file" in st.session_state:
                    try:
                        del st.session_state.audio_file
                    except:
                        pass
        except Exception as e:
            print(f"Error in convert_tts: {str(e)}")

    def load_audio_player(self):
        """
        Display the audio player in the main UI area.
        """
        try:
            # Show audio player in the main area
            if "audio_file" in st.session_state and st.session_state.enable_tts:
                current_audio = st.session_state.audio_file
                st.audio(current_audio, format="audio/mp3", autoplay=True)
            else:
                st.write("No audio available. Click 'Pronounce Word' to hear it.")
        except Exception as e:
            st.error(f"Error displaying audio player: {str(e)}")


# Speech-to-Text (Whisper) class
class WhisperTranscriber:
    """Class for converting speech to text using OpenAI's Whisper model."""

    def __init__(self, api_key=None):
        """Initialize Whisper with OpenAI."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

        # Try to initialize OpenAI client
        try:
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                print("OpenAI client initialized for Whisper transcription")
            else:
                print("OpenAI API key not found for Whisper transcription")
        except Exception as e:
            print(f"Error initializing OpenAI client for Whisper: {str(e)}")

        # Initialize audio key for Streamlit
        if "audio_key" not in st.session_state:
            st.session_state.audio_key = 0

    def get_audio_duration(self, file_path):
        """Get the duration of an audio file."""
        with wave.open(file_path, "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            duration = frames / float(rate)
        return duration

    def crop_audio(self, input_file_path, max_duration=MAX_DURATION_AUDIO):
        """Crop audio to maximum duration."""
        # Create a new temporary file for cropped output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_cropped:
            cropped_path = temp_cropped.name

        # Read and crop from input file
        with wave.open(input_file_path, "rb") as audio:
            params = audio.getparams()
            rate = audio.getframerate()
            frames = audio.getnframes()
            duration = frames / float(rate)

            # If duration is fine, just copy the file
            if duration <= max_duration:
                with wave.open(cropped_path, "wb") as dst_audio:
                    dst_audio.setparams(params)
                    dst_audio.writeframes(audio.readframes(frames))
            else:
                # Crop to max duration
                max_frames = int(rate * max_duration)
                with wave.open(cropped_path, "wb") as dst_audio:
                    st.toast(f"Your recording was cropped to {max_duration} seconds.", icon="🚨")
                    time.sleep(2)
                    dst_audio.setparams(params)
                    dst_audio.writeframes(audio.readframes(max_frames))

        return cropped_path

    def transcribe_audio(self, audio_file):
        """Transcribe audio using Whisper API."""
        if not self.client:
            print("Whisper client not available")
            return None

        try:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            return transcription
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return None

    def process_audio_input(self):
        """Process audio input from microphone."""
        # Display the audio input widget with clear labeling
        st.write("📝 **Say the spelling out loud**")
        audio_file = st.audio_input("Spell the word:",
                                    key=f"audio_input_{st.session_state.audio_key}",
                                    label_visibility="visible")

        if audio_file:
            # Show a spinner while processing
            with st.spinner("Processing your spelling..."):
                input_file_path = None
                cropped_file_path = None

                try:
                    # Save input audio to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_file.write(audio_file.read())
                        input_file_path = temp_file.name

                    # Process the audio and get path to cropped file
                    cropped_file_path = self.crop_audio(input_file_path)

                    # Get duration for logging
                    with wave.open(cropped_file_path, "rb") as audio:
                        frames = audio.getnframes()
                        rate = audio.getframerate()
                        duration = frames / float(rate)
                        print(f"Processed Audio Duration: {duration:.2f} seconds")

                    # Transcribe
                    with open(cropped_file_path, "rb") as audio:
                        transcription = self.transcribe_audio(audio)
                        if transcription:
                            transcription = clean_spelling(transcription)
                            # Clean up and return transcription
                            st.session_state.audio_key += 1
                            st.session_state.spoken_answer = transcription.strip().lower()

                            # Auto-submit after processing if auto-submit is enabled
                            if st.session_state.get("auto_submit_voice", False):
                                # Use a callback to ensure consistent behavior with typing
                                check_spoken_answer()
                                st.rerun()  # Force a rerun to show the feedback

                            return transcription.strip().lower()
                        else:
                            st.error("Failed to transcribe audio. Please try again.")
                            return None

                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                    return None
                finally:
                    # Clean up both temporary files
                    for file_path in [input_file_path, cropped_file_path]:
                        if file_path and os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except PermissionError:
                                pass  # Let OS clean up if file is still locked
        return None


# Initialize TTS engine and STT (Whisper) transcriber
tts_engine = TextToSpeech()
whisper = WhisperTranscriber()

# Initialize session state variables if they don't exist
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'total_attempts' not in st.session_state:
    st.session_state.total_attempts = 0
if 'correct_answers' not in st.session_state:
    st.session_state.correct_answers = 0
if 'incorrect_words' not in st.session_state:
    st.session_state.incorrect_words = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'practice_mode' not in st.session_state:
    st.session_state.practice_mode = "random"
if 'show_word' not in st.session_state:
    st.session_state.show_word = False
if 'voice' not in st.session_state:
    st.session_state.voice = "sage"
if 'input_method' not in st.session_state:
    st.session_state.input_method = "speaking"
if 'spoken_answer' not in st.session_state:
    st.session_state.spoken_answer = ""
if 'auto_submit_voice' not in st.session_state:
    st.session_state.auto_submit_voice = True
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# Sample word list - replace with your own list of spelling words
# This is a simplified random list with various words
WORD_LIST = [
    'abrupt', 'actor', 'attract', 'attraction', 'audiologist', 'audiology', 'audition',
    'auditorium', 'auditory', 'bicycle', 'complete', 'completion', 'compose', 'confer',
    'conservation', 'construct', 'content', 'convertible', 'convivial', 'counteract', 'cycle',
    'deplete', 'depletion', 'descriptive', 'destruct', 'dictate', 'diction', 'dictionary',
    'disagree', 'disappear', 'dispose', 'disrupt', 'dissent', 'diversion', 'edict', 'erupt',
    'export', 'expose', 'extract', 'fertile', 'flexible', 'flexor', 'gradient', 'gradual',
    'graduate', 'graduation', 'import', 'impose', 'indescribable', 'indestructable', 'infertile',
    'inflexible', 'inscribe', 'inscription', 'insensitive', 'instruct', 'interact', 'interrupt',
    'invert', 'local', 'locale', 'locally', 'locate', 'median', 'mediate', 'mediator', 'medium',
    'microscope', 'obstruct', 'oppose', 'periscope', 'plentiful', 'plenty', 'portable', 'predict',
    'prediction', 'preservation', 'preserve', 'propose', 'quietly', 'react', 'reappear', 'recycle',
    'reference', 'reflection', 'reflex', 'repaint', 'reserve', 'reversable', 'revert', 'revival',
    'revive', 'revivify', 'rupture', 'script', 'sensation', 'sensible', 'sensitive', 'sentiment',
    'servant', 'spell', 'structure', 'subservient', 'subtract', 'support', 'teachable', 'telescope',
    'thermal', 'thermocline', 'thermometer', 'thermos', 'thermostat', 'traction', 'tractor',
    'transact', 'transfer', 'transferable', 'transportation', 'tricycle', 'unhappy', 'unicycle',
    'unlock', 'vertical', 'vivacious', 'vivid', 'vividness'
]

# Function to get the appropriate word list
def get_word_list():
    if st.session_state.practice_mode == "incorrect_only" and st.session_state.incorrect_words:
        return st.session_state.incorrect_words
    else:
        return WORD_LIST


def new_word():
    word_list = get_word_list()
    if word_list:
        st.session_state.current_word = random.choice(word_list)
        st.session_state.user_answer = ""
        st.session_state.show_word = False
        st.session_state.spoken_answer = ""

        # Generate audio for the new word
        tts_engine.convert_tts(st.session_state.current_word)
    else:
        st.session_state.current_word = ""


def check_typing_answer():
    user_answer = st.session_state.user_answer.strip().lower()
    check_answer(user_answer)


def check_spoken_answer():
    if st.session_state.spoken_answer:
        user_answer = st.session_state.spoken_answer.strip().lower()
        check_answer(user_answer)


def clean_spelling(text):
    """
    Clean a spelling input by removing spaces, hyphens, punctuation, etc.
    Only keeps alphabetic characters.
    """
    # Remove all non-alphabetic characters and convert to lowercase
    cleaned = ''.join(char for char in text if char.isalpha()).lower()
    return cleaned


def check_answer(user_answer):
    """Process the user's answer and update stats."""
    correct_answer = st.session_state.current_word.lower()

    # Clean both the user answer and the correct answer
    clean_user_answer = clean_spelling(user_answer)
    clean_correct_answer = clean_spelling(correct_answer)

    # Compare the cleaned versions
    is_correct = clean_user_answer == clean_correct_answer

    st.session_state.total_attempts += 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if is_correct:
        st.session_state.correct_answers += 1
        st.session_state.score += 1
        result = "Correct"
        st.session_state.last_result = {"correct": True, "word": correct_answer, "answer": user_answer}
        # Show success message
        st.balloons()
        # If practicing incorrect words and got it right, remove from incorrect list
        if st.session_state.practice_mode == "incorrect_only" and correct_answer in st.session_state.incorrect_words:
            st.session_state.incorrect_words.remove(correct_answer)
    else:
        result = "Incorrect"
        st.session_state.last_result = {"correct": False, "word": correct_answer, "answer": user_answer}
        # Show incorrect message
        st.error(f"The correct spelling was: {correct_answer}")
        if correct_answer not in st.session_state.incorrect_words:
            st.session_state.incorrect_words.append(correct_answer)

    # Record in history
    st.session_state.history.append({
        "timestamp": timestamp,
        "word": correct_answer,
        "user_answer": user_answer,
        "result": result
    })

    # Get a new word
    new_word()


def process_spoken_input():
    """Process the spoken input and update the answer."""
    transcription = whisper.process_audio_input()
    if transcription:
        st.session_state.spoken_answer = transcription
        return transcription
    return None


def reset_stats():
    st.session_state.score = 0
    st.session_state.total_attempts = 0
    st.session_state.correct_answers = 0
    st.session_state.incorrect_words = []
    st.session_state.history = []
    st.session_state.last_result = None
    new_word()


def pronounce_word():
    """Generate and play audio for the current word"""
    tts_engine.convert_tts(st.session_state.current_word)


def toggle_show_word():
    """Toggle whether to show the current word"""
    st.session_state.show_word = not st.session_state.show_word


# Main UI
st.title("🐝 Spelling Bee Practice")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")

    st.session_state.practice_mode = st.radio(
        "Practice Mode",
        options=["random", "incorrect_only"],
        index=0 if st.session_state.practice_mode == "random" else 1,
        help="Random: practice all words. Incorrect only: focus on words you've missed before."
    )

    # Input method selection
    st.session_state.input_method = st.radio(
        "Input Method",
        options=["typing", "speaking"],
        index=0 if st.session_state.input_method == "typing" else 1,
        help="Choose how to enter the spelling: typing or speaking"
    )

    # Auto-submit option for voice input
    if st.session_state.input_method == "speaking":
        st.session_state.auto_submit_voice = st.checkbox(
            "Auto-submit voice answers",
            value=st.session_state.auto_submit_voice,
            help="Automatically submit voice answers after transcription"
        )

    # Voice selection for OpenAI TTS
    st.session_state.voice = st.selectbox(
        "TTS Voice",
        options=["sage", "coral", "alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        index=0,
        help="Select the voice for text-to-speech"
    )

    # TTS toggle
    st.session_state.enable_tts = st.toggle("Enable Voice", value=True)

    if st.button("Reset Statistics"):
        reset_stats()

    # Display statistics
    st.header("Statistics")
    st.write(f"Score: {st.session_state.score}")
    st.write(f"Total Attempts: {st.session_state.total_attempts}")

    if st.session_state.total_attempts > 0:
        accuracy = (st.session_state.correct_answers / st.session_state.total_attempts) * 100
        st.write(f"Accuracy: {accuracy:.1f}%")

    st.write(f"Incorrect Words: {len(st.session_state.incorrect_words)}")
    if st.session_state.incorrect_words:
        st.write("Words to Practice:")
        for word in st.session_state.incorrect_words:
            st.write(f"- {word}")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    # Initialize with a word if needed
    if not st.session_state.current_word:
        new_word()

    # Audio player section
    st.subheader("Listen to the Word")

    # Play button for TTS
    if st.button("🔊 Pronounce Word"):
        pronounce_word()

    # Audio player
    tts_engine.load_audio_player()

    # Different input methods
    if st.session_state.input_method == "typing":
        # Input for typing
        st.text_input(
            "Type the spelling:",
            key="user_answer",
            on_change=check_typing_answer
        )
        st.caption("Press Enter after typing to submit your answer")
    else:
        # Voice input for spelling
        st.subheader("Spell the word by speaking")

        # Display instructions with visual cues
        st.info("📢 Click the microphone, spell the word one letter at a time, then wait for transcription")

        # Process any spoken input
        spoken_answer = process_spoken_input()

        # Show what was heard (if anything)
        if st.session_state.spoken_answer:
            st.success(f"I heard: {st.session_state.spoken_answer}")

            # Submit button for voice input (only show if auto-submit is disabled)
            if not st.session_state.auto_submit_voice:
                if st.button("✓ Submit this spelling", use_container_width=True):
                    check_spoken_answer()

            # Button to try again
            if st.button("🔄 Try again", use_container_width=True):
                st.session_state.spoken_answer = ""
                st.session_state.audio_key += 1
                st.rerun()
        else:
            # More detailed instructions for first-time users
            st.caption("Speak clearly, one letter at a time (example: 'C A T').")

with col2:
    # Quick stats
    if st.session_state.total_attempts > 0:
        accuracy = (st.session_state.correct_answers / st.session_state.total_attempts) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")

    st.metric("Score", st.session_state.score)

# History section
st.header("Practice History")

if not st.session_state.history:
    st.write("No practice history yet. Start practicing!")
else:
    history_df = st.dataframe(
        [{
            "Time": item["timestamp"],
            "Word": item["word"],
            "Your Spelling": item["user_answer"],
            "Result": item["result"]
        } for item in reversed(st.session_state.history)],
        use_container_width=True
    )