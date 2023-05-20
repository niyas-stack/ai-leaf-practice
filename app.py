import streamlit as st
import base64
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Load the model
model = torchvision.models.resnet18(pretrained=True)
classes = {
    0: 'The above leaf is Cassava (Cassava Mosaic)',
    1: 'The above leaf is Cassava CB (Cassava Bacterial Blight)',
    # Rest of the classes...
}

remedies = {
    # Remedies for different diseases...
}

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=(25)),
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
])

def model_predict(image, model_func, transform):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    output = model_func(image_tensor)
    index = torch.argmax(output)
    pred = classes[index.item()]
    probs, _ = torch.max(F.softmax(output, dim=1), 1)
    if probs < 0.93:
        return "not defined", probs
    else:
        return pred, probs

def display_remedies(pred):
    remedy = remedies.get(pred)
    if remedy:
        st.markdown("<p style='color:red;'>Remedy:</p>", unsafe_allow_html=True)
        st.audio(remedy[2], format='audio/mp3')
        st.success(f" {remedy[0]}")

def display_remedies_malayalam(pred):
    remedy = remedies.get(pred)
    if remedy:
        st.markdown("<p style='color:red;'>Remedy (Malayalam):</p>", unsafe_allow_html=True)
        st.audio(remedy[2], format='audio/mp3')
        st.success(f" {remedy[1]}")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    # Set page title and favicon
    st.set_page_config(page_title='My Website', page_icon='logo.png')

    st.markdown('''
    <style>
    .stApp [data-testid="stToolbar"]{
        display:none;
    }
    </style>
    ''', unsafe_allow_html=True)
    add_bg_from_local('background.jpg')

    # Create the header section
    header_container = st.beta_container()

    # Add logo and title to the header
    header_columns = header_container.beta_columns([1, 8])  # Adjust column widths as needed

    with header_columns[0]:
        st.image('logo.png', width=80)

    with header_columns[1]:
        st.title('My Website')

    # Navigation options
    nav_option = st.sidebar.radio('Go to', ('Home', 'About', 'Contact'))

    # Render different content based on the selected navigation option
    if nav_option == 'Home':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width=300)
            st.write("")

            if st.button("Classify", key="classify_btn"):
                pred, probs = model_predict(image, model, transform)
                st.session_state['pred'] = pred
                st.session_state['probs'] = probs
                st.session_state['language_selected'] = False

        if 'pred' in st.session_state and st.session_state['pred'] is not None:
            st.markdown(f"<p style='color: red;'>Prediction: {st.session_state['pred']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: red;'>Probability: {st.session_state['probs']}</p>", unsafe_allow_html=True)

        if 'pred' in st.session_state and st.session_state['pred'] is not None:
            selected_language = st.selectbox("Select Language", ['English', 'Malayalam'], index=0, key="language_select")
            st.session_state['selected_language'] = selected_language

        if 'pred' in st.session_state and st.session_state['pred'] is not None:
            if st.session_state['selected_language'] == 'Malayalam':
                display_remedies_malayalam(st.session_state['pred'])
            else:
                display_remedies(st.session_state['pred'])

    elif nav_option == 'About':
        st.title('About Page')
        # Add content for the about page

    elif nav_option == 'Contact':
        st.title('Contact Page')
        # Add content for the contact page

if __name__ == "__main__":
    main()
