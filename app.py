import streamlit as st

def home_page():
    st.title("Welcome to Dr.Leaf")
    st.write("Instructions:")
    st.write("👉 Take a clear photo of a single leaf.")
    st.write("👉 Ensure that the leaf doesn't have any dust or other unwanted things.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        st.write("")

        if st.button("Classify"):
            pred, probs = model_predict(image, model, transform)
            st.session_state.session_state['pred'] = pred
            st.session_state.session_state['probs'] = probs.item()
            st.session_state.session_state['language_selected'] = False

    if st.session_state.session_state['pred'] is not None:
        st.markdown(f"<p style='color: white;'>Prediction: {st.session_state.session_state['pred']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: white;'>Probability: {st.session_state.session_state['probs']}</p>", unsafe_allow_html=True)
        if st.session_state.session_state['pred'] != 'This is not trained yet':
            selected_language = st.selectbox("Select Language", ['English', 'Malayalam'], index=0, key="language_select")
            st.session_state.session_state['selected_language'] = selected_language

    if st.session_state.session_state['pred'] is not None and not st.session_state.session_state['language_selected']:
        if st.session_state.session_state['selected_language'] == 'Malayalam':
            display_remedies_malayalam(st.session_state.session_state['pred'])
        else:
            display_remedies(st.session_state.session_state['pred'])

def about_page():
    st.title("About Dr.Leaf")
    st.write("Dr.Leaf is an AI-powered application that helps identify plant diseases based on leaf images. It utilizes a deep learning model trained on various plant diseases to provide accurate predictions.")

def contact_page():
    st.title("Contact Dr.Leaf")
    st.write("For any inquiries or support, please contact us at:")
    st.write("- Email: info@drleaf.com")
    st.write("- Phone: 123-456-7890")

# Initialize SessionState
def init_session_state():
    if 'session_state' not in st.session_state:
        st.session_state.session_state = {
            'pred': None,
            'probs': None,
            'selected_language': 'English',
            'language_selected': False
        }

def main():
    init_session_state()

    st.set_page_config(page_title="Dr.Leaf", page_icon="logo.png")
    header_container = st.beta_container()

    # Add logo and title to the header
    header_columns = header_container.beta_columns([1, 8])  # Adjust column widths as needed

    with header_columns[0]:
        st.image('logo.png', width=70)

    with header_columns[1]:
        st.title('Dr.Leaf')

    # Navigation panel
    nav_selection = st.sidebar.radio("Navigation", ["Home", "About", "Contact"])

    if nav_selection == "Home":
        home_page()
    elif nav_selection == "About":
        about_page()
    elif nav_selection == "Contact":
        contact_page()

if __name__ == "__main__":
    main()
