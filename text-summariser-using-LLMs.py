import gradio as gr
import fitz
import re
import easyocr
import numpy as np
from PIL import Image


def preprocess_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()


from huggingface_hub import InferenceClient

client = InferenceClient(
	api_key="YOUR API KEY HERE"
)



def summarize_text(text, prompts):
    cleaned_text = preprocess_text(text)
    combined_text = f"summarize: {cleaned_text}"
    combined_text += " " + " ".join(prompts)
    messages = [
        {
            "role": "user",
            "content":"""Summarize the following text into a short summary: """ + combined_text
        }
    ]

    completion = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
        messages=messages, 
        max_tokens=500,
        temperature= 0.1
    )
    
    return completion.choices[0].message.content

def process_file(file):
    if file is None:
        return None
    
    file_ext = file.name.split('.')[-1].lower()
    
    if file_ext == 'pdf':
        # Convert file content to bytes
        file_content = file.name
        pdf = fitz.open(file_content)  # Open directly with the file path
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()
    elif file_ext == 'txt':
        with open(file.name, 'r', encoding='utf-8') as f:
            text = f.read()
    elif file_ext in ['png', 'jpg', 'jpeg']:
        image = Image.open(file.name)
        reader = easyocr.Reader(['en'])
        result = reader.readtext(np.array(image), detail=0)
        text = ' '.join(result)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    return text

def summarize_interface(
    text_input: str,
    file_input: gr.File,
    prompts: str,
):
    # Process input text
    if file_input is not None:
        text = process_file(file_input)
    else:
        text = text_input
    
    if not text:
        return "No text provided for summarization."
    
    
    prompt_list = [p.strip() for p in prompts.split(',') if p.strip()] if prompts else []
    summary = summarize_text(text, prompt_list)
    
    return summary

# Create Gradio interface
import gradio as gr

css = """
footer {visibility: hidden}
"""

with gr.Blocks(theme=gr.themes.Base(), css=css) as iface:
    gr.Markdown("# Text Summarizer Project")
    gr.Markdown("Upload a file or enter text to generate a summary. Supports PDF, TXT, and image files. Can translate non-English text to English before summarization.")
    
    with gr.Row(equal_height= True):
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter Text",
                placeholder="Enter the text you want to summarize...",
                lines=5
            )
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Or Upload a File (PDF, TXT, or Image)"
            )
    
    with gr.Row(equal_height= True):
        with gr.Column():
            prompts = gr.Textbox(
                label="Prompts (comma-separated)",
                placeholder="Enter additional prompts...",
                lines=2
            )
       
            
    with gr.Row():
        submit_btn = gr.Button(
            "Generate Summary",
            size="lg",
            variant="primary"
        )

    with gr.Row():
        summary_output = gr.Textbox(
            label="Summary Results",
            lines=8,
            show_copy_button=True
        )
    
    # Connect the interface function
    inputs = [text_input, file_input, prompts]
    submit_btn.click(
        fn=summarize_interface,
        inputs=[text_input, file_input, prompts],
        outputs=summary_output
    )

if __name__ == "__main__":
    iface.launch()