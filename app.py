import streamlit as st

# Custom CSS styles
hide_header_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
"""

# Render the custom CSS to hide the default Streamlit header
st.markdown(hide_header_style, unsafe_allow_html=True)

# Render the fixed header
st.markdown(
    """
    <header style="{header_style}">
        <h1>Project Title</h1>
        <img src="logo.png" alt="Logo" style="{logo_style}">
        <nav>
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Contact</a>
        </nav>
    </header>
    """,
    unsafe_allow_html=True
)

# Rest of your Streamlit app code goes here
st.title("Welcome to My Streamlit App")
st.write("This is the content of your app.")


