import gradio as gr
import easyocr
import requests
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import google.generativeai as genai
import os

# Set Google API Key
genai.configure(api_key="AIzaSyBKAHYT6ZQGeRINtqSsO9WnUPb_axUn4n4")  # Replace with your actual key

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-pro")

# EasyOCR Reader
ocr_reader = easyocr.Reader(['en'])

# Fetch content from a URL
def fetch_url_content(url):
    try:
        response = requests.get(url)
        return response.text
    except Exception as e:
        return f"Error fetching URL: {e}"

# Process uploaded files
def process_file(file):
    file_ext = file.name.split('.')[-1].lower()
    file_bytes = BytesIO(file.read())

    if file_ext == "pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text

    elif file_ext in ['jpg', 'jpeg', 'png']:
        img = Image.open(file_bytes)
        ocr_result = ocr_reader.readtext(img)
        text = " ".join([item[1] for item in ocr_result])
        return text

    elif file_ext == "txt":
        return file_bytes.read().decode('utf-8')

    return None

# Summarize using Gemini
def summarize_text(text, summary_style, role, custom_prompt, entity_focus):
    cleaned_text = text.strip()[:3000]  # Truncate to ~3000 chars

    # Build prompt
    prompt_parts = []
    if role:
        prompt_parts.append(f"Summarize this text for a {role}.")
    if summary_style:
        prompt_parts.append(f"Use the style: {summary_style}.")
    if custom_prompt:
        prompt_parts.append(f"Instruction: {custom_prompt}.")
    if entity_focus:
        prompt_parts.append(f"Focus on: {', '.join(entity_focus)}.")

    prompt = "\n".join(prompt_parts)
    final_input = f"{prompt}\n\nText:\n{cleaned_text}"

    try:
        response = model.generate_content(final_input, generation_config={
            "temperature": 0.7,
            "max_output_tokens": 500
        })
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"

# Unified input handler
def generate_summary(text_input, url_input, file_input, summary_style, role, custom_prompt, entity_focus):
    if url_input:
        text = fetch_url_content(url_input)
    elif file_input:
        text = process_file(file_input)
    else:
        text = text_input

    if not text:
        return "No text provided for summarization."

    return summarize_text(text, summary_style, role, custom_prompt, entity_focus)

# Gradio UI
with gr.Blocks() as demo:
    with gr.Column():
        gr.Label("📄 FlexiSummarizer (Gemini)")

        text_input = gr.Textbox(label="Enter Text")
        url_input = gr.Textbox(label="OR Enter a URL")
        file_input = gr.File(file_types=[".pdf", ".txt", ".jpg", ".png"], label="Upload Files")

        summary_style = gr.Dropdown(
            choices=["Simple Summary", "ELI5", "Bullet Points", "Formal Email", "Executive Summary"],
            label="Summary Style",
            value="Simple Summary"
        )

        entity_focus = gr.CheckboxGroup(
            choices=["People", "Organizations", "Dates", "Locations"],
            label="Entity Focus (optional)"
        )

        role_dropdown = gr.Dropdown(
            choices=["People", "Student", "CEO", "Developer", "Lawyer", "Teacher"],
            label="Summarize for Role",
            value="Student"
        )

        custom_prompt = gr.Textbox(label="Optional: Custom Prompt")

        summarize_button = gr.Button("🔍 Summarize")
        summary_output = gr.Textbox(label="📝 Summary Output", lines=10)

    summarize_button.click(
        fn=generate_summary,
        inputs=[text_input, url_input, file_input, summary_style, role_dropdown, custom_prompt, entity_focus],
        outputs=summary_output
    )

demo.launch()
