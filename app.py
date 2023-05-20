import streamlit as st

# Set page title and favicon
st.set_page_config(page_title='My Website', page_icon='logo.png')

st.markdown('''
<style>
.stApp [data-testid="stToolbar"]{
    display:none;
}
.header-container {
    display: flex;
    align-items: center;
}
.header-container img {
    width: 100px;
    margin-right: 20px;
}

@media (max-width: 600px) {
  .header-container {
    flex-direction: row;
    align-items: center;
  }
  .header-container img {
    margin-right: 10px;
  }
}
</style>
''', unsafe_allow_html=True)

# Create the header section
header_container = st.container()

# Add logo and title to the header
with header_container:
    st.image('logo.png', width=100)
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
