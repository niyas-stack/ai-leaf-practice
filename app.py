import streamlit as st

hide_streamlit_style = """
<style>
.stApp > div {
    padding-top: 0rem !important;
}
</style>
"""

st.title("Test")
if st.checkbox('Remove padding'):
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
