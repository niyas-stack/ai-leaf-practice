import streamlit as st

# Add custom CSS styles
header_style = """
    <style>
        .sticky {
            position: fixed;
            top: 0;
            width: 100%;
            background-color: blue;
            padding: 10px;
            z-index: 9999;
        }

        .header-logo {
            width: 100px;
            margin-right: 10px;
        }

        .header-title {
            font-size: 24px;
            font-weight: bold;
        }

        .content {
            margin-top: 100px;
        }
    </style>
"""
st.markdown(header_style, unsafe_allow_html=True)

# Create the sticky header section
header_container = st.beta_container()

# Add logo and title to the header
header_columns = header_container.beta_columns([1, 6])  # Adjust column widths as needed

with header_columns[0]:
    st.image('logo.png', use_column_width=True, caption='Logo', output_format='PNG')

with header_columns[1]:
    st.title('My Website')

# Render the content below the sticky header
st.markdown('<div class="content"></div>', unsafe_allow_html=True)
