import streamlit as st
from navlog.py import add_logo
if st.checkbox("Use url", value=True):
    add_logo("http://placekitten.com/120/120")
else:
    add_logo("gallery/kitty.jpeg", height=300)
st.write("👈 Check out the cat in the nav-bar!")

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
    # Add content for the contact page
