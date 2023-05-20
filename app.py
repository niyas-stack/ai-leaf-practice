import streamlit as st

st.markdown("""
<style>
    #MainMenu, header {visibility: hidden;}
</style>
""",unsafe_allow_html=True)

st.title('This is a test app')

st.sidebar.markdown('After hiding streamlit header there is still a thin white space on top, noticeable when sidebar is open.')

