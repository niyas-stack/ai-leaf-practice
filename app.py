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
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        padding: 10px;
        background-color: #3498DB;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create navigation bar
st.markdown(
    """
    <div class="navbar">
        <div><a href="#">Home</a></div>
        <div><a href="#">About</a></div>
        <div><a href="#">Contact</a></div>
    </div>
    """,
    unsafe_allow_html=True
)

# Rest of your Streamlit app code goes here

