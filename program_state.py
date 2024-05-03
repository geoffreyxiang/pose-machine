from env import *
import cv2
import base64
from image_utils import resize_image
import time

class ProgramState():
  """
  This class helps manage state changes when the pose machine is running, for example freezing frames and tracking which
  subtitles to display on the image
  """
  def __init__(self):
    self.subtitle_text = DEFAULT_START_SUBTITLE
    self.frames_process = None
    self.frame_count = 0
    self.program_running = False
    self.images = []
    self.video_playing = True
    self.freeze_frame_until = None # if not None, we're freezing the frame until this time

    # Sentiment music
    self.sentiment_music_proc = None
    self.sentiment_music_playing = False
    self.chosen_sentiment = "funny"

  def reset_state(self):
    # Kills any running processes, returns variables to defaults
    self.kill_process_frames_if_alive()
    self.kill_sentiment_music_if_alive()
    self.images = []
    self.frame_count = 0
    self.subtitle_text = DEFAULT_START_SUBTITLE

  def reset_state_variables(self):
    self.images = []
    self.frame_count = 0
    self.subtitle_text = DEFAULT_START_SUBTITLE


  def is_program_running(self):
    # Returns if the program has started (ie. after the user starts it)
    return self.program_running

  def is_frame_frozen(self):
    # Returns if we're freezing the frame after taking a photo
    return self.freeze_frame_until != None

  def set_subtitle_text(self, subtitle_text):
    self.subtitle_text = subtitle_text

  def store_image(self, image):
    # image: cv2 frame
    filename = f"frame{len(self.images)}.jpg"
    cv2.imwrite(filename, image)
    self.images.append(image)

  def num_images_stored(self):
    return len(self.images)

  def get_images(self):
    return self.images

  def get_images_as_base86(self):
    # Converts stored images into base64 encoding and returns them
    base86_images = []
    for frame in self.images:
      resized_frame = resize_image(frame)
      _, buffer = cv2.imencode(".jpg", resized_frame)
      base64_image = base64.b64encode(buffer).decode("utf-8")
      base86_images.append(base64_image)
    return base86_images

  def freeze_frame(self, frame):
    # Freezes the webcam frame for three seconds
    self.freeze_frame_until = time.time() + 3, frame.copy()

  def get_frozen_frame(self):
    if self.freeze_frame_until != None:
      return self.freeze_frame_until[1].copy()

  def is_frame_frozen(self):
    return self.freeze_frame_until != None and time.time() < self.freeze_frame_until[0]

  def is_process_frames_running(self):
    # Returns whether the frames are being processed through GPT4 and audio generation
    return self.frames_process != None and self.frames_process.is_alive()

  def kill_process_frames_if_alive(self):
    if self.is_process_frames_running():
      print("Killing process frames")
      self.frames_process.kill()

  def kill_sentiment_music_if_alive(self):
    if self.is_sentiment_music_running():
      print("Killing sentiment music")
      self.sentiment_music_proc.kill()

  def is_sentiment_music_running(self):
    # sentiment music = background music to set the tone
    return self.sentiment_music_proc != None and self.sentiment_music_proc.is_alive()
    # return self.sentiment_music_playing

  def get_sentiment_music_path(self):
    return f"./audio/{self.chosen_sentiment}.mp3"