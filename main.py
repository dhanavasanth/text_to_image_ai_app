import os
from huggingface_hub import InferenceClient
import streamlit as st
from dotenv import load_dotenv

load_dotenv() 

user_prompt = st.text_input("Enter your prompt")

client = InferenceClient(
    provider="replicate",
    api_key=os.getenv('hugging_face_api'),
)
if user_prompt is not None and st.button("Generate"):
    # output is a PIL.Image object
    image = client.text_to_image(
        user_prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
    )

    st.image(image)