import streamlit as st

# Set page title
st.title("My Streamlit App")

# Set page logo
st.image("logo.png", use_column_width=True)

# Set navigation bar
st.markdown(
    """
    <style>
    .navbar {
        display: flex;
        align-items: center;
        background-color: #3498DB;
        color: white;
        padding: 10px 0;
        width: 100%;
    }
    .nav-item {
        margin: 0 10px;
    }
    .nav-item a {
        color: white;
        text-decoration: none;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create navigation bar
st.markdown(
    """
    <div class="navbar">
        <div class="nav-item"><a href="#">Home</a></div>
        <div class="nav-item"><a href="#">About</a></div>
        <div class="nav-item"><a href="#">Contact</a></div>
    </div>
    """,
    unsafe_allow_html=True
)

# Rest of your Streamlit app code goes here

