import streamlit as st

# Predetermined text
predetermined_text = "This is predetermined text."

# Display the text area with predetermined text
text_input = st.text_area( value=predetermined_text, height=200)
