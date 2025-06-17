import logging
import os
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline, AutoTokenizer

# Suppress transformers warnings
logging.basicConfig()
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv(find_dotenv())  # Loading environment variables if you need authentication
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def img2text(url):
    """Convert image to text caption."""
    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    text = image_to_text(url)[0]['generated_text'].strip()
    return text

def generate_story(scenario):
    """Create a vivid short story (under 80 words) from a scenario."""
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    text2text = pipeline("text2text-generation", model="google/flan-t5-small", tokenizer=tokenizer)

    prompt = (
        f"You are a creative storyteller. Provide a vivid, short story (under 80 words) without needless repetition, adding rich details and imagination, based on this scenario:\n"
        f"SCENARIO: {scenario}\nSTORY:"
    )
    story = text2text(prompt, max_new_tokens=80, do_sample=True, temperature=1.2)[0]['generated_text'].strip()
    return story

# Example usage
image_file = "D:\Image-to-story\image.png"

scenario = img2text(image_file)
print(f"Image to text: {scenario}")

story = generate_story(scenario)
print(f"Generated story: {story}")
