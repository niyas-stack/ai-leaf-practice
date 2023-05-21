import streamlit as st

# Set the CSS style for the text area
text_area_style = """
    background-color: black;
    color: white;
"""

# Add the text area with the modified style
text_input = st.text_area("Enter your text", value="", height=200, style=text_area_style)

# You can continue with the rest of your Streamlit app code
