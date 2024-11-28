import os
from datetime import datetime

import gradio as gr
import pandas as pd
from llama_cpp import Llama
import copy

model_name = "Taiwar/llama-3.2-1b-instruct-lora_model-1epoch"

model = Llama.from_pretrained(
    repo_id=model_name,
    filename="llama-3.2-1b-instruct-lora_merged-1epoch-16b.gguf",
    # verbose=False,
    # chat_format="llama-3"
)

hf_token = os.getenv('hf_token')
if hf_token is None:
    print("Reading hf_token from .hftoken file")
    hf_token = open(".hftoken").read().strip()


def load_context():
    _aq_predictions = pd.read_csv("data/aq_predictions.csv")
    _aq_predictions['date'] = pd.to_datetime(_aq_predictions['date']).dt.date
    return _aq_predictions

aq_predictions = load_context()

def get_aq_prediction(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d').date()
    prediction = aq_predictions[aq_predictions['date'] == date]
    if not prediction.empty:
        return prediction.to_dict(orient='records')[0]
    else:
        return None

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    # Extract date from the message (assuming the date is provided in the message)
    import re
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', message)
    if date_match:
        print("Matched date:", date_match.group(0))
        date = date_match.group(0)
        aq_prediction = get_aq_prediction(date)
        system_message += f"\nYou were asked about the air quality prediction for {date}"
        if aq_prediction:
            print("Air quality prediction for", date, ":", aq_prediction)
            system_message += f"\nThe air quality prediction for Reutlingen, Germany on {date}: {aq_prediction}"
        else:
            system_message += f"\nNo air quality prediction available for {date}"

    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # _ = model.generate(
    #     input_ids=inputs,
    #     streamer=text_streamer,
    #     max_new_tokens=max_tokens,
    #     use_cache=True,
    #     temperature=temperature,
    #     top_p=top_p,
    # )

    print("Messages:", messages)

    text_streamer  = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    response = ""
    for token in text_streamer:
        print("Model response:", token)
        if "text" in token["choices"][0]:
            response += token["choices"][0]["text"]
        else:
            response += str(token["choices"][0])
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

if __name__ == "__main__":
    demo.launch()