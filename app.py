import streamlit as st
import base64

def add_sticky_header():
    st.markdown(
        """
        <style>
        .sticky {
            position: sticky;
            top: 0;
            z-index: 100;
            background-color: yellow;
            padding: 10px;
        }
        </style>
        """
        , unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="sticky">
            <!-- Add your title and logo here -->
            <h1>AI Leaf Disease Detection</h1>
        </div>
        """
        , unsafe_allow_html=True
    )

def main():
    add_sticky_header()

    # Rest of your Streamlit app code
    st.title("Welcome to my app")
    st.write("This is the content of my app.")

if __name__ == "__main__":
    main()
