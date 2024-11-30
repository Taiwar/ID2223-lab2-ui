import os
from datetime import datetime

import gradio as gr
import hopsworks
import pandas as pd
from llama_cpp import Llama
from openai import OpenAI


def read_api_credentials():
    credentials = {}
    for key in ['model_api_key', 'model_api_url', 'hf_token', 'hw_token', 'do_hw_query']:
        credentials[key] = os.getenv(key)
        if credentials[key] is None:
            print(f"Reading {key} from .{key} file")
            credentials[key] = open(f".{key}").read().strip()
    return credentials['model_api_key'], credentials['model_api_url'], credentials['hf_token'], credentials['hw_token'], credentials['do_hw_query']

model_api_key, model_api_url, hf_token, hw_token, do_hw_query = read_api_credentials()
os.environ["HOPSWORKS_API_KEY"] = hw_token

def load_context():
    _aq_predictions = pd.read_csv("data/aq_predictions.csv")
    _aq_predictions['date'] = pd.to_datetime(_aq_predictions['date']).dt.date
    return _aq_predictions


# Init first model option: OpenAI API client for deployment running on modal with GPU
model_name_1 = "llama-3.2-1b-instruct-lora-1poch_merged16b"
client = OpenAI(
    base_url=model_api_url,
    api_key=model_api_key
)

# Init second model option: Running it locally with CPU
model_name_2 = "Taiwar/llama-3.2-1b-instruct-lora_model-1epoch"

model = Llama.from_pretrained(
    repo_id=model_name_2,
    filename="llama-3.2-1b-instruct-lora_merged-1epoch-16b.gguf",
    verbose=False,
    # chat_format="llama-3"
)

project = hopsworks.login()
fs = project.get_feature_store()
predictions_fg = fs.get_feature_group("aq_predictions")

cache = {}

def get_aq_predictions(backfill_days):
    # Get predictions from yesterday to the future
    today = datetime.now().date()
    backfill_day = today - pd.Timedelta(days=backfill_days)
    if backfill_day in cache:
        return cache[backfill_day]
    predictions_df = predictions_fg.filter(predictions_fg.date >= backfill_day).read()
    # For each date, there are multiple predictions. We only want the most recent one (lowest days_before_forecast_day)
    predictions_df = predictions_df.sort_values("days_before_forecast_day").groupby("date").first().reset_index()
    cache[backfill_day] = predictions_df
    return predictions_df

def format_predictions_for_llm(predictions_df):
    formatted_rows = []
    for _, row in predictions_df.iterrows():
        formatted_row = (
            f"Date: {row['date']}, "
            f"Temperature: {row['temperature_2m_mean']}°C, "
            f"Precipitation: {row['precipitation_sum']}mm, "
            f"Wind Speed: {row['wind_speed_10m_max']}m/s, "
            f"Wind Direction: {row['wind_direction_10m_dominant']}°, "
            f"City: {row['city']}, "
            f"Lagged: {row['lagged']}, "
            f"Predicted PM2.5: {row['predicted_pm25']}µg/m³, "
            f"Street: {row['street']}, "
            f"Country: {row['country']}, "
            # f"Days Before Forecast: {row['days_before_forecast_day']}"
        )
        formatted_rows.append(formatted_row)
    return "\n".join(formatted_rows)


def respond(
    message,
    history: list[tuple[str, str]],
    model_type,
    max_tokens,
    temperature,
    top_p,
    repeat_penalty,
    backfill_days
):
    system_message = "\nYou are a friendly Chatbot."
    system_message += (
        "\nYou help summarize and answer questions about air quality predictions."
        "\nAnswer any questions about this data helpfully."
        "\nDo not make predictions about dates not provided, only answer based on the data provided in the system messages."
    )
    if do_hw_query:
        aq_prediction = get_aq_predictions(backfill_days)
        system_message += f"The following air quality data was retrieved: \n{format_predictions_for_llm(aq_prediction)}"
    else:
        aq_prediction = load_context()
        system_message += f"The following air quality data was loaded from the cache: \n{format_predictions_for_llm(aq_prediction)}"

    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    if model_type == "local":
        text_streamer  = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            repeat_penalty=repeat_penalty
        )

        response = ""
        for token in text_streamer:
            if "delta" in token["choices"][0]:
                delta = token["choices"][0]["delta"]
                if "content" in delta:
                    response += delta["content"]
            else:
                print("Unexpected token:", token)
            yield response
    elif model_type == "remote":
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name_1,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=repeat_penalty
        )
        response = chat_completion.choices[0].message.content
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Dropdown(
            choices=["local", "remote"],
            value="local",
            label="Model type"
        ),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.0,
            step=0.1,
            label="Repeat penalty",
        ),
        gr.Slider(
            minimum=1,
            maximum=30,
            value=1,
            step=1,
            label="Backfill days"
        )
    ],
)

if __name__ == "__main__":
    demo.launch()