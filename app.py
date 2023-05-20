import streamlit as st

enable_scroll = """
<style>
.main {
    overflow: auto;
}
</style>
"""

st.markdown(enable_scroll, unsafe_allow_html=True)

st.title("Test")
if st.checkbox('Remove padding'):
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
