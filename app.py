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
.logo {
    width: 100px;
    margin-right: 20px;
}
</style>
''', unsafe_allow_html=True)

# Create the header section
header_container = st.beta_container()

# Add logo and title to the header
header_columns = header_container.beta_columns([1, 6])  # Adjust column widths as needed

with header_columns[0]:
    st.image('logo.png', use_column_width=False, caption='Logo', width=100, class_='logo')

with header_columns[1]:
    st.title('My Website')

# Navigation options and content rendering code...


