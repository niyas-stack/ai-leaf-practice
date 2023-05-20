import streamlit as st

# Set page title and favicon
st.set_page_config(page_title='My Website', page_icon='logo.png')

# Add custom CSS styles
header_style = """
    <style>
        .header {
            position: sticky;
            top: 0;
            left: 0;
            width: 100%;
            height: 80px;
            background-color: yellow;
            z-index: 9999;
            padding: 10px;
            display: flex;
            align-items: center;
        }

        .header-logo {
            width: 100px;
            margin-right: 10px;
        }

        .header-title {
            font-size: 24px;
            font-weight: bold;
        }
    </style>
"""
st.markdown(header_style, unsafe_allow_html=True)

# Create the fixed header section
header_container = st.beta_container()

# Add logo and title to the header
header_columns = header_container.beta_columns([1, 6])  # Adjust column widths as needed

with header_columns[0]:
    st.image('logo.png', use_column_width=True, caption='Logo', output_format='PNG')

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

