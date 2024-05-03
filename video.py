"""
Manages the video creation process after the audio voiceovers have been generated
"""

from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from moviepy.video.compositing.transitions import fadein, fadeout
import cv2
from ftfy import fix_text
import time
from pydub import AudioSegment

import os
from elevenlabs import generate, set_api_key
from dotenv import load_dotenv
from env import *

def make_clip(image_path, audio_path, text, use_audio):
    # Creates an image clip with audio voiceover

    # initialize audio clip
    audio_clip = AudioFileClip(audio_path, fps=22050)

    # edit image to have text
    image_with_text_path = f'annotated_frame.jpg'
    image = add_text_to_image(image_path, text, image_with_text_path)

    # initialize image clip with duration of audio clip
    image_clip = ImageClip(image_with_text_path)
    if use_audio:
        print(audio_path)
        image_clip = image_clip.set_duration(audio_clip.duration)
        image_clip = image_clip.set_audio(audio_clip)
    else:
        # default to 15 seconds if we are not using audio
        image_clip = image_clip.set_duration(15)
    return image_clip

def add_text_to_image(image_path, text, save_as, max_line_length=30):
    # Superimposes the provided text on the image and saves it to the `save_as` file path
    image = cv2.imread(image_path)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color for the main text
    shadow_color = (0, 0, 0)  # Black color for the shadow
    line_type = 2
    margin = 10  # Margin for text from the bottom
    line_spacing = 30  # Space between lines
    shadow_offset = 2  # Offset for shadow

    # Split text into multiple lines
    text = fix_text(text)
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

    cv2.imwrite(save_as, image)
    return image

def make_and_show_video(image_paths, audio_paths, subtitles, use_audio=True, fade_duration=0.5, background_music_path = None):
    # Creates a video using the provided images and audio voiceovers. Then displays it for the user
    clips = [make_clip(img, aud, text, use_audio) for img, aud, text in zip(image_paths, audio_paths, subtitles)]

    # apply crossfade effect to clips
    faded_clips = [fadeout(clips[0], duration=fade_duration)] + [fadeout(fadein(clip, duration=fade_duration), duration=fade_duration) for clip in clips[1:]]
    final_clip = concatenate_videoclips(faded_clips, method="compose")

    if background_music_path != None:
        if final_clip.audio != None:
            final_clip.audio.write_audiofile("final_audio.mp3", fps = 22050)
            merge_audio_files(background_music_path, "final_audio.mp3") # writes to another audio file
            mixed_audio = AudioFileClip("./mixed_audio.wav", fps = 22050)
        else:
            mixed_audio = AudioFileClip(background_music_path, fps = 22050)
        final_duration = final_clip.duration
        final_clip = final_clip.set_audio(mixed_audio)
        final_clip = final_clip.set_duration(final_duration)

    # show the final video
    final_clip.preview()

    return final_clip

def merge_audio_files(audio_path1, audio_path2):
    # Load the two audio clips
    clip1 = AudioSegment.from_file(audio_path1)
    clip2 = AudioSegment.from_file(audio_path2)

    # Ensure both audio clips have the same sample rate
    clip1 = clip1.set_frame_rate(22050)
    clip2 = clip2.set_frame_rate(22050)

    # Ensure both audio clips have the same number of channels
    clip1 = clip1.set_channels(2)
    clip2 = clip2.set_channels(2)

    # Adjust the volume of each clip if necessary (optional)
    clip1 = clip1 - 15  # Increase volume of clip1 by 6 dB
    # clip2 = clip2 - 3  # Decrease volume of clip2 by 3 dB

    # Mix the two audio clips together
    mixed_clip = clip1.overlay(clip2)

    # Export the mixed audio clip to a file
    mixed_clip.export("mixed_audio.wav", format="wav")