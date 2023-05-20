import streamlit as st

st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}

    /* This code gets the first element on the sidebar,
    and overrides its default styling */
    section[data-testid="stSidebar"] div:first-child {
        top: 0;
        height: 100vh;
    }
</style>
""",unsafe_allow_html=True)

st.title('This is a test app')

st.sidebar.markdown('After hiding streamlit header there is still a thin white space on top, noticeable when sidebar is open.')

