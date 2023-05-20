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

# Add custom CSS for the navigation bar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: url('logo.png');
        background-repeat: no-repeat;
        background-position: 20px 20px;
        padding-top: 120px; /* Adjust the padding as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

