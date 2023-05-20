import streamlit as st

def main():
    # Add custom CSS styles to create a sticky header
    st.markdown(
        """
        <style>
        .sticky {
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 20px;
            background-color: #F8D700;
            color: black;
        }
        .navbar-logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: bold;
        }
        .navbar-logo img {
            width: 40px;
            height: 40px;
        }
        .navbar-menu {
            display: flex;
            align-items: center;
            gap: 20px;
            list-style-type: none;
            margin: 0;
        }
        .navbar-menu li {
            padding: 5px;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create the sticky header
    st.beta_container().markdown(
        """
        <div class="sticky">
            <div class="navbar">
                <div class="navbar-logo">
                    <img src="logo.png" alt="Logo">
                    <h1>Project Title</h1>
                </div>
                <div class="navbar-menu">
                    <li>Home</li>
                    <li>About</li>
                    <li>Contact</li>
                </div>
                <div class="navbar-hamburger">
                    <!-- Add your hamburger menu icon here -->
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add your Streamlit app content below the sticky header
    st.title('Welcome to my Streamlit App')
    st.write('This is the content of your app.')

if __name__ == '__main__':
    main()
