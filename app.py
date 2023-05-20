import streamlit as st

hide_streamlit_style = """
<style>
.css-hi6a2p {padding-top: 0rem;}
</style>

"""
st.title("Test")
if st.checkbox('Remove padding'):
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
