"""
Utility functions for managing function calls to GPT4
"""

import requests
import json
from env import *

api_key = OPENAI_API_KEY

def get_sentiment_prompt(sentiment):
    with open('./prompts.json', 'r') as file:
        data = json.load(file)
        return data[sentiment]["prompt"]

def get_sentiment_voice_id(sentiment):
    with open('./prompts.json', 'r') as file:
        data = json.load(file)
        return data[sentiment]["voice_id"]

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
There should be 4 chunks to this story, 1 per image. Limit each chunk to 40 words. Don't use the word image. In your response, you should only have the 4 paragraphs. Don't use a dash or em dash in your sentences.
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
    print(f'chosen sentiment: {sentiment}')
    return gpt_4_output

def parse_gpt4_response(text):
    res = text.split('\n\n')
    # assert len(res) == 4, f'{len(res)} chunks detected instead of 4'
    return res