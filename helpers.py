import os
import re
import tempfile
import uuid
import time
from pathlib import Path
from time import time
from typing import Optional, Tuple


# Import OpenAI for text-to-speech and speech-to-text
import streamlit as st
from openai import OpenAI


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

