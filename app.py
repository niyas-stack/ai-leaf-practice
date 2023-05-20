import streamlit as st

# Set page title
st.title("My Streamlit App")

# Add logo and title to sidebar
st.sidebar.image('logo.png', use_column_width=True)
st.sidebar.title('Navigation')

# Create the navigation bar
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

