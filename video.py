from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from moviepy.video.compositing.transitions import fadein, fadeout
import cv2

# testing
import os
from elevenlabs import generate, set_api_key
from dotenv import load_dotenv
from env import *
from ftfy import fix_text

# load_dotenv()
# set_api_key(os.environ.get("ELEVENLABS_API_KEY"))

def make_clip(image_path, audio_path, text, use_audio):
    # initialize audio clip
    audio_clip = AudioFileClip(audio_path, fps=22050)

    # edit image to have text
    image_with_text_path = f'annotated_{image_path}'
    add_text_to_image(image_path, text, image_with_text_path)

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

def make_video(image_paths, audio_paths, subtitles, use_audio=True, fade_duration=0.5):
    clips = [make_clip(img, aud, text, use_audio) for img, aud, text in zip(image_paths, audio_paths, subtitles)]

    # apply crossfade effect to clips
    faded_clips = [fadeout(clips[0], duration=fade_duration)] + [fadeout(fadein(clip, duration=fade_duration), duration=fade_duration) for clip in clips[1:]]
    final_clip = concatenate_videoclips(faded_clips, method="compose")

    # write final video
    final_clip.write_videofile("story.mp4", fps=24)

subtitles = ['In a room filled with energy, a young man flashes a peace sign with a confident smile, setting off on a spirited adventure.', 'The upbeat tempo rises as he switches to one hand, his eyes gleaming with determination and readiness for what’s to come.', 'Caught in a lively rhythm, he ruffles his hair, his expression playful yet focused, hinting at an exciting challenge ahead.', 'The crescendo peaks as he triumphantly holds up a small, victorious object. His journey culminates in this small yet significant triumph, reflected in his satisfied smirk.']

# generate audio files
# for i in range(len(subtitles)):
#     audio = generate(subtitles[i], voice=os.environ.get("ELEVENLABS_VOICE_ID"))

#     with open(f'audio{i}.wav', "wb") as f:
#         print('generating for file:', i)
#         f.write(audio)

# programmatically create and save a video
make_video([f'frame{i}.jpg' for i in range(4)], [f'audio{i}.wav' for i in range(4)], subtitles, use_audio=False)