import base64
import os
from multiprocessing import Process, Queue, Pipe

import cv2
import numpy as np
import pygame
import requests
import time
# from dotenv import load_dotenv
from elevenlabs import generate, play, set_api_key
from queue_objects import QueueEvent, QueueEventTypes
import json
import random
from video import make_video
from moviepy.editor import VideoFileClip
from gpt_utils import pass_to_gpt4_vision, parse_gpt4_response
from image_utils import resize_image, add_subtitle

from env import *

# load_dotenv()
api_key = OPENAI_API_KEY

# set_api_key(os.environ.get("ELEVENLABS_API_KEY"))
set_api_key(ELEVENLABS_API_KEY)


def play_music(track_path, loop = True, volume = 0.2):
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

def spawn_process_and_play_video(video_path):
    video_process = Process(target=play_video, args=(video_path, ))
    video_process.start()

    return video_process

def webcam_capture(queue):
    chosen_sentiment = "funny" # Default value, will be changed later

    # cap = cv2.VideoCapture("./example_vid.avi")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    cv2.namedWindow("Pose Machine", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pose Machine", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Pose Machine", 98, 0)

    subtitle_text = DEFAULT_START_SUBTITLE
    process_frames_running = False
    frames_process = None
    frame_count = 0
    program_running = False
    images = []
    video_playing = True
    freeze_frame_until = None # if not None, we're freezing the frame until this time

    # Sentiment music
    sentiment_music_proc = None
    sentiment_music_playing = False

    # create pipe for communicating between webcam (parent) and process_frame (child) processes
    parent_conn, child_conn = Pipe()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        if program_running:
            if len(images) < 4:
                if freeze_frame_until != None:
                    subtitle_text = "Captured! Taking another photo soon..."
                elif frame_count % CAPTURE_EVERY_X_FRAMES == 0:
                    # Play a camera click sound effect
                    spawn_process_and_play("./camera_click.mp3", loop = False, volume = 1)
                    print("----capturing----")

                    filename = f"frame{len(images)}.jpg"
                    cv2.imwrite(filename, frame)

                    # Process the image for inputting into GPT4V
                    resized_frame = resize_image(frame)
                    retval, buffer = cv2.imencode(".jpg", resized_frame)
                    base64_image = base64.b64encode(buffer).decode("utf-8")

                    images.append(base64_image)

                    # Freeze the frame to show the user what image will be used
                    freeze_frame_until = time.time() + 3, frame.copy()
                else:
                    subtitle_text = "Pose!"
            elif len(images) >= 4 and not process_frames_running:
                # Spin off new process to run gpt4 processing in the background
                print("[Main process] processing images")
                print("[Main process] Have sent the images!")

                subtitle_text = "The Pose Machine will be back shortly..."

                frames_process = Process(target=process_frames, args=(child_conn, chosen_sentiment))
                frames_process.start()

                parent_conn.send((QueueEventTypes.PROCESS_IMAGES, { "images": images }))

        else:
            # Reset all variables
            subtitle_text = DEFAULT_START_SUBTITLE
            images = []

        sentiment_music_playing = sentiment_music_proc != None and sentiment_music_proc.is_alive()
        if not sentiment_music_playing and not program_running: # when we finish a cycle, we should select a new sentiment
            chosen_sentiment = random.choice(SENTIMENTS)
            print(f"Chosen sentiment {chosen_sentiment}")

            # Background music for the chosen sentiment
            sentiment_music_proc = spawn_process_and_play(f"{chosen_sentiment}.mp3")

        # We use this to check if frame processing (child process) is done, in which case we'll reset the program
        if parent_conn.poll():
            event, payload = parent_conn.recv()
            print(event)

            if event == QueueEventTypes.PROCESSING_DONE:
                if sentiment_music_playing:
                    sentiment_music_proc.kill()
                # video_process = spawn_process_and_play_video(payload["video_file"])
                # video_process.join() # wait until video is done before continuing
            elif event == QueueEventTypes.VIDEO_DONE:
                if sentiment_music_playing:
                    sentiment_music_proc.kill()
                program_running = False

                # TODO: Spawn another process to write the video to slack

        process_frames_running = frames_process != None and frames_process.is_alive()

        if freeze_frame_until != None:
            frame = freeze_frame_until[1].copy()
            if time.time() > freeze_frame_until[0]:
                freeze_frame_until = None

        frame_with_subtitle = add_subtitle(frame, frame_count, subtitle_text, show_countdown=(program_running and not process_frames_running))
        cv2.imshow("Pose Machine", frame_with_subtitle)

        if not program_running and cv2.waitKey(1) & 0xFF == ord("s"):
            program_running = True
            frame_count = 1
        elif program_running and cv2.waitKey(1) & 0xFF == ord("r"):
            # Restart and go to program start
            program_running = False
            freeze_frame_until = None
            if process_frames_running:
                frames_process.kill()
            if sentiment_music_playing:
                sentiment_music_proc.kill()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            if process_frames_running:
                frames_process.kill()
            if sentiment_music_playing:
                sentiment_music_proc.kill()
            break

        if freeze_frame_until == None:
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def process_frames(conn, chosen_sentiment):
    if not conn.poll(5000):
        return

    event, payload = conn.recv()
    if event != QueueEventTypes.PROCESS_IMAGES: # other queue events are unhandled
        return

    # print(f"Received payload {payload}")

    # payload should be a list of base64 encoded images to pass into gpt4 vision
    gpt_4_output = pass_to_gpt4_vision(payload["images"], chosen_sentiment)

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
    except:
        # programmatically create and save a video without audio (likely ran out of elevenlabs tokens)
        conn.send((QueueEventTypes.PROCESSING_DONE, {}))
        print("error generating audio files. likely ran out of elevenlabs tokens. creating video without audio...")
        audio_success = False
        make_video([f'frame{i}.jpg' for i in range(4)], [f'audio{i}.wav' for i in range(4)], subtitles, use_audio=False, background_music_path=f"{chosen_sentiment}.mp3")

    # programmatically create and save a video with audio
    if audio_success:
        conn.send((QueueEventTypes.PROCESSING_DONE, {}))
        make_video([f'frame{i}.jpg' for i in range(4)], [f'audio{i}.wav' for i in range(4)], subtitles, background_music_path=f"{chosen_sentiment}.mp3")

    conn.send((QueueEventTypes.VIDEO_DONE, {} ))

def main():
    queue = Queue()
    webcam_capture(queue)
    # webcam_process = Process(target=webcam_capture, args=(queue,))
    # # frames_process = Process(target=process_frames, args=(queue,))
    # music_proc = Process(target=play_music, args=("./camera_click.mp3", False))
    # spawn_process_and_play_video("./story.mp4")
    # spawn_process_and_play("./camera_click.mp3", loop = False, volume = 1)

    # webcam_process.start()
    # # frames_process.start()
    # music_proc.start()

    # webcam_process.join()
    # frames_process.join()
    # music_proc.join()


if __name__ == "__main__":
    main()
