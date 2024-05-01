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

from env import OPENAI_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

# load_dotenv()
api_key = OPENAI_API_KEY

# set_api_key(os.environ.get("ELEVENLABS_API_KEY"))
set_api_key(ELEVENLABS_API_KEY)

CAPTURE_TIME_BUFFER = 3
CAPTURE_EVERY_X_FRAMES = 30 * CAPTURE_TIME_BUFFER
DEFAULT_START_SUBTITLE = "Hold 's' to begin"
SENTIMENTS = ["happy", "sad", "epic"]

def play_music(track_path):
    print(f"Playing track {track_path}")
    # Initialize pygame mixer
    pygame.mixer.init()

    sound = pygame.mixer.Sound(track_path)
    sound.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # You can adjust the tick rate as needed
    # Load the music file
    # pygame.mixer.music.load(track_path)
    # pygame.mixer.music.set_volume(0.3)
    # # Play the music file indefinitely (the argument -1 means looping forever)
    # pygame.mixer.music.play()

# Process target function
def music_process():
    play_music("exit.mp3")  # Replace with your actual file path


# new function for passing all images and sentiment to gpt4 vision
def pass_to_gpt4_vision(base64_images, sentiment):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "system",
                "content": ("""
The user will submit 4 images. Use the first image to serve as the introduction. Craft an introduction to the characters in the image. Use the next two images to serve as the body of the story.
Narrate each image individually, but make a coherent storyline throughout. Finally, use the last image to make a satisfying conclusion.
You may be creative with interpreting the images, but ensure that the characters and gestures depicted are accurate. Do not give any characters names. The characters must be nameless.
There should be 4 chunks to this story, 1 per image. Limit each chunk to 40 words. Don't use the word image.
          """ + get_sentiment_prompt(sentiment)).strip(),
            },
        ]
        + format_images_for_gpt4_vision(base64_images),
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    gpt_4_output = response.json()["choices"][0]["message"]["content"]
    return gpt_4_output

def generate_new_line(base64_image):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this scene like you're a narrator in a movie",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ]


def resize_image(image, max_width=500):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the ratio of the width and apply it to the new width
    ratio = max_width / float(width)
    new_height = int(height * ratio)

    # Resize the image
    resized_image = cv2.resize(
        image, (max_width, new_height), interpolation=cv2.INTER_AREA
    )
    return resized_image


def add_subtitle(image, frame_count, text="", show_countdown = True, max_line_length=40):
    countdown_seconds = CAPTURE_TIME_BUFFER - (frame_count % CAPTURE_EVERY_X_FRAMES) // 30  # Convert frame count to seconds

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color for the main text
    shadow_color = (0, 0, 0)  # Black color for the shadow
    line_type = 2
    margin = 10  # Margin for text from the bottom
    line_spacing = 30  # Space between lines
    shadow_offset = 2  # Offset for shadow

    # Split text into multiple lines
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + word) <= max_line_length:
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)  # Add the last line

    # Calculate the starting y position
    text_height_total = line_spacing * len(lines)
    start_y = image.shape[0] - text_height_total - margin

    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, line_type)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = start_y + i * line_spacing

        # Draw shadow
        cv2.putText(
            image,
            line,
            (text_x + shadow_offset, text_y + shadow_offset),
            font,
            font_scale,
            shadow_color,
            line_type,
        )

        # Draw main text
        cv2.putText(
            image, line, (text_x, text_y), font, font_scale, font_color, line_type
        )

    font_scale = 2
    countdown_text = f"{countdown_seconds}"
    text_size = cv2.getTextSize(countdown_text, font, font_scale, line_type)[0]
    text_x = image.shape[1] - text_size[0] - margin  # Position to the right
    text_y = margin + text_size[1]  # Position at the top

    if show_countdown:
        # Draw shadow for countdown
        cv2.putText(image, countdown_text, (text_x + shadow_offset, text_y + shadow_offset), font, font_scale, shadow_color, line_type)

        # Draw countdown text
        cv2.putText(image, countdown_text, (text_x, text_y), font, font_scale, font_color, line_type)

    return image

def get_sentiment_prompt(sentiment):
    with open('./prompts.json', 'r') as file:
        data = json.load(file)
        return data[sentiment]["prompt"]

def parse_gpt4_response(text):
    res = text.split('\n\n')
    assert len(res) == 4, f'{len(res)} chunks detected instead of 4'
    return res

def format_images_for_gpt4_vision(base64_images):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                for base64_image in base64_images
            ]
        }
    ]

def spawn_process_and_play(audio_file):
    music_process = Process(target=play_music, args=(audio_file,))
    music_process.start()

def play_video(video_path):
    clip = VideoFileClip(video_path)
    clip.preview()

def spawn_process_and_play_video(video_path):
    video_process = Process(target=play_video, args=(video_path, ))
    video_process.start()

def webcam_capture(queue):
    # cap = cv2.VideoCapture("./example_vid.avi")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Webcam", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

    subtitle_text = DEFAULT_START_SUBTITLE
    process_frames_running = False
    frames_process = None
    frame_count = 0
    program_running = False
    images = []
    video_playing = True
    freeze_frame_until = None # if not None, we're freezing the frame until this time

    # video = cv2.VideoWriter("./example_vid.avi", 0, 1, (1280, 720))

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

                    # Play a camera click sound effect
                    spawn_process_and_play("./audio0.wav")
                else:
                    subtitle_text = "Pose!"
            elif len(images) >= 4 and not process_frames_running:
                # Spin off new process to run gpt4 processing in the background
                print("[Main process] processing images")
                subtitle_text = "Processing!"
                print("[Main process] Have sent the images!")

                frames_process = Process(target=process_frames, args=(child_conn,))
                frames_process.start()

                parent_conn.send((QueueEventTypes.PROCESS_IMAGES, { "images": images }))

        else:
            # Reset all variables
            subtitle_text = DEFAULT_START_SUBTITLE
            images = []

        # We use this to check if frame processing (child process) is done, in which case we'll reset the program
        if parent_conn.poll():
            event, payload = parent_conn.recv()
            print(event)

            if event == QueueEventTypes.PROCESSING_DONE:
                # cap.release()
                # cap = cv2.VideoCapture(payload["video_file"])
                spawn_process_and_play_video(payload["video_file"])
                program_running = False

        process_frames_running = frames_process != None and frames_process.is_alive()
        # frame = enhance_image_contrast_saturation(frame)

        if freeze_frame_until != None:
            frame = freeze_frame_until[1].copy()
            if time.time() > freeze_frame_until[0]:
                freeze_frame_until = None

        frame_with_subtitle = add_subtitle(frame, frame_count, subtitle_text, show_countdown=(program_running and not process_frames_running))
        cv2.imshow("Webcam", frame_with_subtitle)
        # print(frame.shape)


        if not program_running and cv2.waitKey(1) & 0xFF == ord("s"):
            program_running = True
            frame_count = 1
        elif program_running and cv2.waitKey(1) & 0xFF == ord("r"):
            # Restart and go to program start
            program_running = False
            freeze_frame_until = None
            if process_frames_running:
                frames_process.kill()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if freeze_frame_until == None:
            frame_count += 1

    cap.release()
    # video.release()
    cv2.destroyAllWindows()


def process_frames(conn):
    # if not conn.poll(): # the queue should contain the image list
    #     return
    if not conn.poll(5000):
        return

    event, payload = conn.recv()
    if event != QueueEventTypes.PROCESS_IMAGES: # other queue events are unhandled
        return

    # print(f"Received payload {payload}")

    # payload should be a list of base64 encoded images to pass into gpt4 vision
    gpt_4_output = pass_to_gpt4_vision(payload["images"], random.choice(SENTIMENTS))

    subtitles = parse_gpt4_response(gpt_4_output)
    print(subtitles)

    # generate audio files
    audio_success = True
    try: 
        for i in range(len(subtitles)):
            audio = generate(subtitles[i], voice=ELEVENLABS_VOICE_ID)

            with open(f'audio{i}.wav', "wb") as f:
                print('generating audio for file:', i)
                f.write(audio)
    except:
        # programmatically create and save a video without audio (likely ran out of elevenlabs tokens)
        print("error generating audio files. likely ran out of elevenlabs tokens. creating video without audio...")
        audio_success = False
        make_video([f'frame{i}.jpg' for i in range(4)], [f'audio{i}.wav' for i in range(4)], subtitles, use_audio=False)
        
    # programmatically create and save a video with audio
    if audio_success:
        make_video([f'frame{i}.jpg' for i in range(4)], [f'audio{i}.wav' for i in range(4)], subtitles, use_audio=True)

    conn.send((QueueEventTypes.PROCESSING_DONE, { "video_file": "./story.mp4" }))

    # cap = cv2.VideoCapture(0)

    # if not cap.isOpened():
    #     print("Error: Webcam not accessible in process_frames.")
    #     return

    # frame_count = 0
    # script = []
    # pic_count = 0
    # is_processing = False
    # while True:
    #     ret, frame = cap.read()
    #     frame = cv2.flip(frame, 1)

    #     if not ret:
    #         break

    #     if pic_count < 4 and frame_count % CAPTURE_EVERY_X_FRAMES == 0:
    #         print("----capturing----")

    #         filename = f"frame{pic_count}.jpg"
    #         cv2.imwrite(filename, frame)

    #         queue.put((QueueEventTypes.FREEZE_FRAME, { "freeze_time": 3 }))

    #         pic_count += 1
    #     elif pic_count < 4 and frame_count % CAPTURE_EVERY_X_FRAMES != 0:
    #         queue.put((QueueEventTypes.SHOW_SUBTITLE, { "subtitle": "Pose!" }))

    #     elif pic_count >= 4 and not is_processing:
    #         is_processing = True
    #         queue.put((QueueEventTypes.SHOW_SUBTITLE, { "subtitle": "Processing" }))
    #         resized_frame = resize_image(frame)
    #         # retval, buffer = cv2.imencode(".jpg", resized_frame)
    #         # base64_image = base64.b64encode(buffer).decode("utf-8")
    #         # gpt_4_output = pass_to_gpt4_vision(base64_image, script)
    #         # script = script + [{"role": "assistant", "content": gpt_4_output}]
    #         # print("script:", script)

    #         queue.put((QueueEventTypes.KILL_FRAME_PROCESSING, {}))

    #     frame_count += 1
    #     # play_audio(gpt_4_output)
    #     # time.sleep()  # Wait for 1 second

    # cap.release()

def main():
    queue = Queue()
    webcam_process = Process(target=webcam_capture, args=(queue,))
    # frames_process = Process(target=process_frames, args=(queue,))
    # music_proc = Process(target=music_process)
    # spawn_process_and_play_video("./story.mp4")

    webcam_process.start()
    # frames_process.start()
    # music_proc.start()

    webcam_process.join()
    # frames_process.join()
    # music_proc.join()


if __name__ == "__main__":
    main()
