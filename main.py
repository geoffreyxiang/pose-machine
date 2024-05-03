from multiprocessing import Process, Pipe

import cv2
import pygame
import requests
from elevenlabs import generate, set_api_key
from queue_objects import QueueEventTypes
import json
import random
from video import make_and_show_video
from gpt_utils import pass_to_gpt4_vision, parse_gpt4_response, get_sentiment_voice_id
from image_utils import  add_subtitle
from program_state import ProgramState

from env import *

api_key = OPENAI_API_KEY

set_api_key(ELEVENLABS_API_KEY)

# Plays music from a given sound track either infinitely or only once (sound effects)
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

# Spawns a new process to play an audio file
def spawn_process_and_play(audio_file, loop = True, volume = 0.2):
    music_process = Process(target=play_music, args=(audio_file, loop, volume))
    music_process.start()
    return music_process

# Main entry point for application - controls webcam input
def webcam_capture():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    cv2.namedWindow("Pose Machine", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pose Machine", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Pose Machine", 98, 0)

    state = ProgramState()

    # create pipe for communicating between webcam (parent) and process_frame (child) processes
    parent_conn, child_conn = Pipe()

    while True:
        if not state.is_frame_frozen():
            ret, frame = cap.read() # reads a frame from the web cam
            frame = cv2.flip(frame, 1)
            if not ret:
                break
        else:
            frame = state.get_frozen_frame().copy() # if the frame is in a frozen state, we simply get the saved frame

        if state.is_program_running():
            if state.num_images_stored() < 4:
                if state.is_frame_frozen():
                    state.subtitle_text = "Captured! Taking another photo soon..."
                elif state.frame_count % CAPTURE_EVERY_X_FRAMES == 0:
                    # Play a camera click sound effect
                    spawn_process_and_play("./camera_click.mp3", loop = False, volume = 1)
                    print("----capturing----")

                    # Store the frame and save a copy to local directory
                    state.store_image(frame)

                    # Freeze the frame to show the user what image will be used
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
            state.reset_state_variables()

        # when we finish a cycle, we should select a new sentiment
        if not state.is_sentiment_music_running() and not state.is_program_running():
            state.chosen_sentiment = random.choice(SENTIMENTS)
            print(f"Chosen sentiment {state.chosen_sentiment}")

            # Background music for the chosen sentiment
            state.sentiment_music_proc = spawn_process_and_play(state.get_sentiment_music_path())

        # We use this to check if frame processing (child process) is done, in which case we'll reset the program
        if parent_conn.poll():
            event, payload = parent_conn.recv()
            print(event)

            if event == QueueEventTypes.PROCESSING_DONE:
                state.kill_sentiment_music_if_alive()

            elif event == QueueEventTypes.VIDEO_DONE:
                state.reset_state_variables()
                state.kill_sentiment_music_if_alive()
                state.program_running = False


        # Adds subtitles to the frame shown to the user
        frame_with_subtitle = add_subtitle(frame, state.frame_count, state.subtitle_text, show_countdown=(state.is_program_running() and not state.is_process_frames_running()))
        cv2.imshow("Pose Machine", frame_with_subtitle)

        if not state.is_program_running() and cv2.waitKey(1) & 0xFF == ord("s"):
            state.program_running = True
            state.frame_count = 1
        elif state.is_program_running() and cv2.waitKey(1) & 0xFF == ord("r"):
            # Restart and go to program start
            state.reset_state()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            state.reset_state()
            break

        if not state.is_frame_frozen():
            state.frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Function used for a process to create the story for the stored frames
# It will also call other functions to create a video based on the GPT-generated story
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
