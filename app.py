import streamlit as st

# Custom CSS styles
header_style = """
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background-color: yellow;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    z-index: 999;
"""

logo_style = """
    height: 40px;
    width: auto;
"""

# Hide Streamlit default styles
hide_menu_style = """
    <style>
    #Header{ visibility: hidden; }
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Render the fixed header
st.markdown(
    f"""
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

