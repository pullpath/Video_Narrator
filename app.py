import streamlit as st
import requests
import base64
from dotenv import load_dotenv
import os
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import tempfile
import cv2
from openai import OpenAI
import io
import numpy as np

load_dotenv()
 
def main():
    st.set_page_config(page_title="Video Narrator", page_icon=":dog:")

    st.header("Video Narrator :dog: ")
    uploaded_file = st.file_uploader("Choose a video...")

    if uploaded_file is not None:
        st.video(uploaded_file)
        prompt = st.text_area("Prompt", value="These are frames of a video. Create a short voiceover script that can be used along this video. No extra text is needed in your response, just the script itself.")
        
        voice = st.selectbox(
            "Pick the voice for your generated video",
            ("alloy", "echo", "fable", "onyx", "nova", "shimmer"),
            index=0,
            placeholder="Pick a voice"
        )


    if st.button('Generate', type="primary") and uploaded_file is not None:
        with st.spinner('Processing...'):
            # Comment out the following section, which used to be put all logics in python based ai server

            # url = "http://localhost:8000/api/video_narrator"
            # response = requests.post(url, files={'video_file': uploaded_file}, data={'prompt': prompt})
            # json_result = response.json()['result']
            # st.video(json_result['video'])

            # End of the commented section
            base64Frames, video_filename, video_duration  = video_to_frame(uploaded_file)
            est_word_count = np.ceil(video_duration * 2.5)
            final_prompt = prompt + f"(This video is ONLY {video_duration} seconds long, so make sure the voice over MUST be able to be explained in less than {est_word_count} words)"
            print(final_prompt)
            video_script = frames_to_story(base64Frames=base64Frames, prompt=final_prompt)

            st.write(video_script)
            # Generate audio from text
            audio_filename, audio_bytes_io = text_to_audio(video_script, voice)

            # Merge audio and video
            output_video_filename = os.path.splitext(video_filename)[0] + '_output.mp4'
            final_video_filename = merge_audio_video(video_filename, audio_filename, output_video_filename)

            # Display the result
            st.video(final_video_filename)

            # Clean up the temporary files
            os.unlink(video_filename)
            os.unlink(audio_filename)
            os.unlink(final_video_filename)


def video_to_frame(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(video_file.read())
        video_filename = tmpfile.name

    video_duration = VideoFileClip(video_filename).duration

    video = cv2.VideoCapture(video_filename)
    base64_frames = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        base64_frames.append(base64.b64encode(buffer).decode('utf-8'))

    video.release()
    print(len(base64_frames), "frames read.")
    return base64_frames, video_filename, video_duration

def frames_to_story(base64Frames, prompt):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"),
        default_headers={"x-pp-token": os.environ.get("X-PP-TOKEN")},
    )
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
                *map(lambda x: {"image": x, "resize": 768},
                     base64Frames[0::25]),
            ],
        },
    ]
    result = client.chat.completions.create(
        messages=PROMPT_MESSAGES,
        model="gpt-4o-mini",
        max_tokens=500,
    )
    return result.choices[0].message.content

def text_to_audio(text, voice):
    response = requests.post(
        f"{os.environ['OPENAI_API_BASE']}audio/speech",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "x-pp-token": os.environ.get("X-PP-TOKEN"),
        },
        json={
            "model": "tts-1",
            "input": text,
            "voice": voice,
        },
    )

    # audio_file_path = "output_audio.wav"
    # with open(audio_file_path, "wb") as audio_file:
    #     for chunk in response.iter_content(chunk_size=1024 * 1024):
    #         audio_file.write(chunk)

    # # To play the audio in Jupyter after saving
    # Audio(audio_file_path)
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception("Request failed with status code")
    # ...
    # Create an in-memory bytes buffer
    audio_bytes_io = io.BytesIO()

    # Write audio data to the in-memory bytes buffer
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio_bytes_io.write(chunk)

    # Important: Seek to the start of the BytesIO buffer before returning
    audio_bytes_io.seek(0)

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            tmpfile.write(chunk)
        audio_filename = tmpfile.name

    return audio_filename, audio_bytes_io

def merge_audio_video(video_filename, audio_filename, output_filename):
    print("Merging audio and video...")
    print("Video filename:", video_filename)
    print("Audio filename:", audio_filename)

    # Load the video file
    video_clip = VideoFileClip(video_filename)

    # Load the audio file
    audio_clip = AudioFileClip(audio_filename)

    # Set the audio of the video clip as the audio file
    final_clip = video_clip.set_audio(audio_clip)

    # Write the result to a file (without audio)
    final_clip.write_videofile(
        output_filename, codec='libx264', audio_codec='aac', threads=14)

    # Close the clips
    video_clip.close()
    audio_clip.close()

    print('Output filename:', output_filename)
    # Return the path to the new video file
    return output_filename

if __name__ == "__main__":
    main()