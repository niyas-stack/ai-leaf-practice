import streamlit as st

# Set page title and favicon
st.set_page_config(page_title='My Website', page_icon='logo.png')

# Create the navigation bar
st.sidebar.title('Navigation')

# Add logo and title to the navigation bar
st.sidebar.image('logo.png', use_column_width=True)
st.sidebar.title('My Website')

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
