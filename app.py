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
        
        .menu-toggle {
            display: none;
        }
        
        .menu-toggle-label {
            display: none;
            cursor: pointer;
        }
        
        .menu-toggle-label span {
            display: inline-block;
            width: 20px;
            height: 2px;
            background-color: black;
            margin-bottom: 4px;
        }
        
        .menu-toggle:checked ~ #menu {
            display: block;
        }
        
        #menu {
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            z-index: 100;
            background-color: yellow;
            padding: 10px;
            width: 100%;
        }
        
        #menu a {
            display: block;
            margin-bottom: 10px;
        }
        
        @media (max-width: 768px) {
            .menu-toggle-label {
                display: block;
            }
            
            .menu-toggle:checked + .menu-toggle-label span:nth-child(2) {
                opacity: 0;
            }
            
            .menu-toggle:checked + .menu-toggle-label span:nth-child(3) {
                transform: rotate(-45deg) translate(-6px, 4px);
            }
            
            .menu-toggle:checked + .menu-toggle-label span:nth-child(4) {
                transform: rotate(45deg) translate(-6px, -4px);
            }
            
            #menu {
                position: relative;
                top: auto;
                left: auto;
                width: auto;
                background-color: transparent;
                padding: 0;
            }
            
            #menu a {
                margin-bottom: 0;
            }
        }
        </style>
        """
        , unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <input class="menu-toggle" type="checkbox" id="menu-toggle">
        <label class="menu-toggle-label" for="menu-toggle">
            <span></span>
            <span></span>
            <span></span>
        </label>
        
        <div class="sticky">
            <!-- Add your title and logo here -->
            <h1>AI Leaf Disease Detection</h1>
            <div id="menu">
                <a href="#">Home</a>
                <a href="#">About</a>
                <a href="#">Contact</a>
            </div>
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
