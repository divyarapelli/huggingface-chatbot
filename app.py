import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

st.set_page_config(page_title="🤖 Chat with DialoGPT")
st.title("🤖 Chatbot using DialoGPT-medium")
st.markdown("Start a conversation with an AI model trained for chatting!")

# Load model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model = model.to("cpu")

# Session state to maintain chat history
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_user_inputs" not in st.session_state:
    st.session_state.past_user_inputs = []
if "past_bot_outputs" not in st.session_state:
    st.session_state.past_bot_outputs = []

# Text input from user
user_input = st.text_input("You:", key="input")

if user_input:
    # Encode the user input + append EOS
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to("cpu")

    # Append to previous history if exists
    bot_input_ids = (
        torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
        if st.session_state.chat_history_ids is not None
        else new_input_ids
    )

    # Generate a response with sampling
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,        # adds randomness
        top_p=0.9,              # nucleus sampling
        do_sample=True          # enables sampling
    )

    # Decode and extract only the new response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Update session state
    st.session_state.chat_history_ids = chat_history_ids
    st.session_state.past_user_inputs.append(user_input)
    st.session_state.past_bot_outputs.append(response)

# Show full conversation history
for user_text, bot_reply in zip(st.session_state.past_user_inputs, st.session_state.past_bot_outputs):
    st.write(f"🧑 You: {user_text}")
    st.write(f"🤖 Bot: {bot_reply}")

# Reset button
if st.button("🔄 Reset Chat"):
    st.session_state.chat_history_ids = None
    st.session_state.past_user_inputs = []
    st.session_state.past_bot_outputs = []
