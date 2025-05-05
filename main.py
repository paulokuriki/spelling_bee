import streamlit as st
import random
from datetime import datetime
from helpers import TextToSpeech


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




# Initialize TTS engine
tts_engine = TextToSpeech()


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
if 'voice' not in st.session_state:
    st.session_state.voice = "sage"
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'used_words' not in st.session_state:
    st.session_state.used_words = []

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

# Add this to your initial session state setup



def new_word():
    # Create a list of available words (those not yet used)
    available_words = [word for word in WORD_LIST if word not in st.session_state.used_words]

    # If all words have been used, reset the used_words list
    if not available_words:
        st.session_state.used_words = []
        available_words = WORD_LIST
        st.success("Great job! You've completed all words. Starting a new round!")

    # Choose a random word from available words
    st.session_state.current_word = random.choice(available_words)
    st.session_state.used_words.append(st.session_state.current_word)
    st.session_state.user_answer = ""

    # Generate audio for the new word
    tts_engine.convert_tts(st.session_state.current_word)


def check_typing_answer():
    user_answer = st.session_state.user_answer.strip().lower()
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




def reset_stats():
    st.session_state.score = 0
    st.session_state.total_attempts = 0
    st.session_state.correct_answers = 0
    st.session_state.incorrect_words = []
    st.session_state.history = []
    st.session_state.last_result = None
    st.session_state.used_words = []
    new_word()


def pronounce_word():
    """Generate and play audio for the current word"""
    try:
        tts_engine.convert_tts(st.session_state.current_word)
    except Exception as e:
        st.error(f"Could not pronounce word: {e}")



# Main UI
st.title("🐝 Spelling Bee Practice")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")

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

    # Input for typing
    st.text_input(
        "Type the spelling:",
        key="user_answer",
        on_change=check_typing_answer
    )
    st.caption("Press Enter after typing to submit your answer")

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