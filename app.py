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

st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}

    /* This code gets the first element on the sidebar,
    and overrides its default styling */
    section[data-testid="stSidebar"] div:first-child {
        top: 0;
        height: 100vh;
    }
</style>
""", unsafe_allow_html=True)

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

st.title('This is a test app')

st.sidebar.markdown('After hiding streamlit header there is still a thin white space on top, noticeable when the sidebar is open.')

