import base64
import os
from multiprocessing import Process, Queue, Pipe

import cv2
import numpy as np
import pygame
import requests
import time
from dotenv import load_dotenv
from elevenlabs import generate, set_api_key
from queue_objects import QueueEventTypes
import json
import random
from video import make_and_show_video
from moviepy.editor import VideoFileClip
from gpt_utils import pass_to_gpt4_vision, parse_gpt4_response
from image_utils import resize_image, add_subtitle
from program_state import ProgramState

from env import *

load_dotenv()
api_key = OPENAI_API_KEY

# set_api_key(os.environ.get("ELEVENLABS_API_KEY"))
set_api_key(ELEVENLABS_API_KEY)


def play_music(track_path, loop = True, volume = 0.2):
    """
    Plays music according to the provided track path. Note that this music playing is blocking,
    so you should probably spin up a new process.
    """
    print(f"Playing track {track_path}")
    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the music file
    pygame.mixer.music.load(track_path)
    pygame.mixer.music.set_volume(volume)
    if loop:
        # Play the music file indefinitely (the argument -1 means looping forever)
        pygame.mixer.music.play(-1)
    else:
        pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # You can adjust the tick rate as needed

def get_sentiment_voice_id(sentiment):
    with open('./prompts.json', 'r') as file:
        data = json.load(file)
        return data[sentiment]["voice_id"]


def spawn_process_and_play(audio_file, loop = True, volume = 0.2):
    music_process = Process(target=play_music, args=(audio_file, loop, volume))
    music_process.start()
    return music_process

def play_video(video_path):
    clip = VideoFileClip(video_path)
    clip.preview()

# Create a separate process to
def spawn_process_and_play_video(video_path):
    video_process = Process(target=play_video, args=(video_path, ))
    video_process.start()

    return video_process

def webcam_capture():
    # chosen_sentiment = "funny" # Default value, will be changed later

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    cv2.namedWindow("Pose Machine", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pose Machine", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Pose Machine", 98, 0)

    state = ProgramState()

    # subtitle_text = DEFAULT_START_SUBTITLE
    # process_frames_running = False
    # frames_process = None
    # frame_count = 0
    # program_running = False
    # images = []
    # video_playing = True
    # freeze_frame_until = None # if not None, we're freezing the frame until this time

    # # Sentiment music
    # sentiment_music_proc = None
    # sentiment_music_playing = False

    # create pipe for communicating between webcam (parent) and process_frame (child) processes
    parent_conn, child_conn = Pipe()

    while True:
        if not state.is_frame_frozen():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
        else:
            frame = state.get_frozen_frame().copy()

        if state.is_program_running():
            if state.num_images_stored() < 4:
                if state.is_frame_frozen():
                    state.subtitle_text = "Captured! Taking another photo soon..."
                elif state.frame_count % CAPTURE_EVERY_X_FRAMES == 0:
                    # Play a camera click sound effect
                    spawn_process_and_play("./camera_click.mp3", loop = False, volume = 1)
                    print("----capturing----")

                    state.store_image(frame)
                    # Write the frame to disk to use in the video production
                    # filename = f"./tmp/frame{len(images)}.jpg"
                    # cv2.imwrite(filename, frame)

                    # Convert the image into base86 and save it for inputting it into an LLM later
                    # resized_frame = resize_image(frame)
                    # retval, buffer = cv2.imencode(".jpg", resized_frame)
                    # base64_image = base64.b64encode(buffer).decode("utf-8")

                    # images.append(base64_image)

                    # Freeze the frame to show the user what image will be used
                    # freeze_frame_until = time.time() + 3, frame.copy()
                    state.freeze_frame(frame)
                else:
                    state.subtitle_text = "Pose!"
            elif state.num_images_stored() >= 4 and not state.is_process_frames_running():
                # Spin off new process to run gpt4 processing in the background
                print("[Main process] processing images")

                state.subtitle_text = "The Pose Machine will be back shortly..."

                state.frames_process = Process(target=process_frames, args=(child_conn, state.chosen_sentiment))
                state.frames_process.start()

                parent_conn.send((QueueEventTypes.PROCESS_IMAGES, { "base86_images": state.get_images_as_base86() }))
                print("[Main process] Have sent the images!")

        else:
            # Reset all variables
            # subtitle_text = DEFAULT_START_SUBTITLE
            # images = []
            state.reset_state_variables()

        # sentiment_music_playing = state.sentiment_music_proc != None and state.sentiment_music_proc.is_alive()
        if not state.is_sentiment_music_running() and not state.is_program_running(): # when we finish a cycle, we should select a new sentiment
            state.chosen_sentiment = random.choice(SENTIMENTS)
            print(f"Chosen sentiment {state.chosen_sentiment}")

            # Background music for the chosen sentiment
            state.sentiment_music_proc = spawn_process_and_play(state.get_sentiment_music_path())
            # state.sentiment_music_proc.start()
            # state.sentiment_music_proc.join()

        # We use this to check if frame processing (child process) is done, in which case we'll reset the program
        if parent_conn.poll():
            event, payload = parent_conn.recv()
            print(event)

            if event == QueueEventTypes.PROCESSING_DONE:
                state.kill_sentiment_music_if_alive()
                # if sentiment_music_playing:
                #     sentiment_music_proc.kill()
                # video_process = spawn_process_and_play_video(payload["video_file"])
                # video_process.join() # wait until video is done before continuing
            elif event == QueueEventTypes.VIDEO_DONE:
                state.reset_state_variables()
                state.kill_sentiment_music_if_alive()
                state.program_running = False
                # if sentiment_music_playing:
                #     sentiment_music_proc.kill()
                # program_running = False
                # images = []

        # process_frames_running = frames_process != None and frames_process.is_alive()

        # if freeze_frame_until != None:
        #     frame = freeze_frame_until[1].copy()
        #     if time.time() > freeze_frame_until[0]:
        #         freeze_frame_until = None

        frame_with_subtitle = add_subtitle(frame, state.frame_count, state.subtitle_text, show_countdown=(state.is_program_running() and not state.is_process_frames_running()))
        cv2.imshow("Pose Machine", frame_with_subtitle)

        if not state.is_program_running() and cv2.waitKey(1) & 0xFF == ord("s"):
            state.program_running = True
            state.frame_count = 1
        elif state.is_program_running() and cv2.waitKey(1) & 0xFF == ord("r"):
            # Restart and go to program start
            state.reset_state()
            # state.program_running = False
            # freeze_frame_until = None
            # if process_frames_running:
            #     frames_process.kill()
            # if sentiment_music_playing:
            #     sentiment_music_proc.kill()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            # if process_frames_running:
            #     frames_process.kill()
            # if sentiment_music_playing:
            #     sentiment_music_proc.kill()
            state.reset_state()
            break

        if not state.is_frame_frozen():
            state.frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def process_frames(conn, chosen_sentiment):
    if not conn.poll(5000):
        return

    event, payload = conn.recv()
    if event != QueueEventTypes.PROCESS_IMAGES: # other queue events are unhandled
        return

    # payload should be a list of base64 encoded images to pass into gpt4 vision
    gpt_4_output = pass_to_gpt4_vision(payload["base86_images"], chosen_sentiment)

    subtitles = parse_gpt4_response(gpt_4_output)
    print(subtitles)

    # generate audio files
    audio_success = True
    try:
        for i in range(len(subtitles)):
            audio = generate(subtitles[i], voice=get_sentiment_voice_id(chosen_sentiment))

            with open(f'audio{i}.wav', "wb") as f:
                print('generating audio for file:', i)
                f.write(audio)
    except Exception as e:
        # programmatically create and save a video without audio (likely ran out of elevenlabs tokens)
        conn.send((QueueEventTypes.PROCESSING_DONE, {}))
        print(e)
        print("error generating audio files. likely ran out of elevenlabs tokens. creating video without audio...")
        audio_success = False
        video = make_and_show_video([f"frame{i}.jpg" for i in range(4)], [f'audio{i}.wav' for i in range(4)], subtitles, use_audio=False, background_music_path=f"{chosen_sentiment}.mp3")

    # programmatically create and save a video with audio
    if audio_success:
        conn.send((QueueEventTypes.PROCESSING_DONE, {}))
        video = make_and_show_video([f"frame{i}.jpg" for i in range(4)], [f'audio{i}.wav' for i in range(4)], subtitles, background_music_path=f"{chosen_sentiment}.mp3")

    conn.send((QueueEventTypes.VIDEO_DONE, { } ))

    video.write_videofile("story.mp4", fps=24)
    # send_to_slack("story.mp4")

def save_videofile(video):
    video.write_videofile("story.mp4", fps=24)

def send_to_slack(video_file):
    """
    Video_file: file path to a video file to be sent to our slack channel
    """
    token = SLACK_TOKEN
    channels = 'C071KLK6E11' # Channel ID where you want to upload the video
    file_name = 'video.mp4'

    with open(video_file, 'rb') as file_content:
        payload = {
            'channels': channels,
            'filename': file_name,
            'title': 'Story',
            }
        files = {
            'file': file_content,
        }
        headers = {
            'Authorization': f'Bearer {token}'
        }
        response = requests.post('https://slack.com/api/files.upload', headers=headers, data=payload, files=files)

        print(response.text) # To see the response from Slack


def main():
    webcam_capture()

if __name__ == "__main__":
    main()
