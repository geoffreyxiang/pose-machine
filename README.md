[Demo](https://www.loom.com/share/9b16ac22576c480ea324b1a617c3215e?sid=b035920d-6ff0-4204-8486-1f23deff7316)

"Pose Machine" invites strangers to strike four independent poses and crafts a coherent video that weaves together the poses in a story. Each video comes with a voiceover, background music, and the pose images themselves; the content of the story depends on a randomly chosen sentiment (ex. happy, funny).

To get this running, install the dependencies and execute `python3 main.py`.

There are a few dependencies to install like openai and elevenlabs.

You'll also need to create a `.env` file like so:

```
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
```

Inspiration and original code taken from https://github.com/farzaa/ai-movie
