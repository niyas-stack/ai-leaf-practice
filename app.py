import streamlit as st

# Set page title and favicon
st.set_page_config(page_title='My Website', page_icon='logo.png')

# Create the header section
header_container = st.beta_container()

# Add logo and title to the header
header_columns = header_container.beta_columns([1, 6])  # Adjust column widths as needed

with header_columns[0]:
    st.image('logo.png', use_column_width=True)

with header_columns[1]:
    st.title('My Website')

# Navigation options
nav_option = st.sidebar.radio('Go to', ('Home', 'About', 'Contact'))

# Render different content based on the selected navigation option
if nav_option == 'Home':
    st.title('Home Page')
    # Add content for the home page
elif nav_option == 'About':
    st.title('About Page')
    # Add content for the about page
elif nav_option == 'Contact':
    st.title('Contact Page')
    # Add content for the contact page

# Custom CSS styles
header_style = """
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background-color: white;
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

# Render custom CSS styles
st.markdown(
    f"""
    <style>
    /* Hide Streamlit header */
    .stAppHeader {{
        display: none;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

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

