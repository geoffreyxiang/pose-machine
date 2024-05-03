from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
ELEVENLABS_API_KEY=os.environ.get("ELEVENLABS_API_KEY")
SLACK_TOKEN=os.environ.get("SLACK_TOKEN")
CAPTURE_TIME_BUFFER = 5
CAPTURE_EVERY_X_FRAMES = 30 * CAPTURE_TIME_BUFFER
DEFAULT_START_SUBTITLE = "Hold 's' to begin"
SENTIMENTS = ["happy", "epic", "funny", "mysterious"]