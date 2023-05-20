import streamlit as st

# Custom CSS styles
custom_css = """
<style>
body {
    margin: 0;
    padding: 0;
}

.stApp {
    background-color: transparent;
}

.stButton>button {
    background-color: transparent;
    color: inherit;
}

.stButton>button:hover {
    background-color: transparent;
}

.stTextInput>div>div {
    background-color: transparent;
}

.stTextInput>div>input {
    color: inherit;
}

.stMarkdown {
    background-color: transparent;
    color: inherit;
}

.stMarkdown p {
    margin: 0;
    padding: 0;
}

.stMarkdown a {
    color: inherit;
}

.stMarkdown a:hover {
    text-decoration: underline;
}
</style>
"""

# Render the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

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




