import os
import numpy as np
from torchsummary import summary
import torch
import PIL
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import base64


# Load the model
model = torchvision.models.resnet18(pretrained=True)
classes = {
    0: 'The above leaf is Cassava (Cassava Mosaic)',
    1: 'The above leaf is Cassava CB (Cassava Bacterial Blight)',
    2: 'The above leaf is Cassava Healthy leaf',
    3: 'The above leaf is Tomato Bacterial spot',
    4: 'The above leaf is Tomato early blight',
    5: 'The above leaf is Tomato Late blight',
    6: 'The above leaf is Tomato Leaf Mold',
    7: 'The above leaf is Tomato Septoria leaf spot',
    8: 'The above leaf is Tomato Spider mites Two-spotted spider mite',
    9: 'The above leaf is Tomato Target Spot',
    10: 'The above leaf is Tomato Yellow Leaf Curl Virus',
    11: 'The above leaf is Tomato mosaic virus',
    12: 'The above leaf is Tomato healthy',
    13: 'The above leaf is bean angular leaf spot',
    14: 'The above leaf is bean healthy',
    15: 'The above leaf is bean rust'
}
remedies = {
    'The above leaf is Cassava (Cassava Mosaic)': [
        'Remedy for Cassava Mosaic', 'കാസവ മോസായികയുടെ പരിഹാരം',
        'cassava.m4a', 'cassava.m4a'
    ],
    'The above leaf is Cassava CB (Cassava Bacterial Blight)': [
        'Remedy for Cassava Bacterial Blight', 'കാസവ ബാക്ടീരിയൽ ബ്ലൈറ്റിന്റെ പരിഹാരം',
        'cassava.m4a', 'cassava.m4a'
    ]
    # add remedies for other diseases in both English and Malayalam
}

selected_language = 'English'  # Set the default language


num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model_path = "epoch-90.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
summary(model, input_size=(3, 224, 224))
model.eval()

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

def add_navigation_bar():
    st.markdown(
        """
        <style>
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #f8f9fa;
            padding: 10px;
        }

        .navbar-logo {
            display: flex;
            align-items: center;
        }

        .navbar-logo img {
            margin-right: 10px;
        }

        .navbar-title {
            font-size: 24px;
            font-weight: bold;
            color: #000;
        }

        .navbar-menu {
            display: flex;
        }

        .navbar-menu-item {
            margin-right: 20px;
        }

        .navbar-menu-item:last-child {
            margin-right: 0;
        }
        </style>
        """
        "<div class='navbar'>"
        "<div class='navbar-logo'>"
        "<img src='logo.png' width='50'>"
        "<span class='navbar-title'>AI Leaf Disease Detection</span>"
        "</div>"
        "<div class='navbar-menu'>"
        "<div class='navbar-menu-item'><a href='#' class='navbar-link'>Home</a></div>"
        "<div class='navbar-menu-item'><a href='#' class='navbar-link'>About</a></div>"
        "<div class='navbar-menu-item'><a href='#' class='navbar-link'>Contact</a></div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True
    )
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


def add_bg_from_local(image_file):
    file_extension = os.path.splitext(image_file)[1].lower()
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()

    if file_extension == ".jpg" or file_extension == ".jpeg":
        image_type = "jpeg"
    elif file_extension == ".png":
        image_type = "png"
    else:
        raise ValueError("Unsupported image file format. Only JPEG and PNG are supported.")

    st.markdown(
        f"""
        <style>
        @media (max-width: 768px) {{
            .stApp {{
                background-image: url(data:image/{image_type};base64,{encoded_string});
                background-size: contain;
                backdrop-filter: blur(25px);
            }}
        }}
        @media (min-width: 769px) {{
            .stApp {{
                background-image: url(data:image/{image_type};base64,{encoded_string});
                background-repeat: no-repeat;
                background-position: center;
                background-size: cover;
                backdrop-filter: blur(25px);
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
def display_remedies(pred):
    remedy = remedies.get(pred)
    if remedy:
        st.markdown("<p style='color:red;'>Remedy:</p>", unsafe_allow_html=True)
        if selected_language == 'English':
            audio_file = remedy[2]
        else:
            audio_file = remedy[3]
        with open(audio_file, 'rb') as audio:
            st.audio(audio.read(), format='audio/mp3')
        if selected_language == 'English':
            st.success(f" {remedy[0]}")
        else:
            st.success(f" {remedy[1]}")


def display_remedies_malayalam(pred):
    remedy = remedies.get(pred)
    if remedy:
        st.markdown("<p style='color:red;'>Remedy (Malayalam):</p>", unsafe_allow_html=True)
        audio_file = remedy[3]
        with open(audio_file, 'rb') as audio:
            st.audio(audio.read(), format='audio/mp3')
        st.success(f" {remedy[1]}")


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

    st.set_page_config(page_title="AI Leaf Disease Detection", page_icon=":leaves:")
    add_navigation_bar()
    st.markdown(
        """
        <style>
        .title-wrapper {
            display: flex;
            align-items: center;
        }
        .title-wrapper img {
            margin-left: 10px;
        }
        .logo-wrapper {
            display: flex;
            justify-content: flex-end;
            margin-top: -80px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="title-wrapper">
            <h1 style='color: green; font-family: Playfair Display;'>AI Leaf Disease Detection</h1>
            <img src="logo.png"  width="50">
        </div>
        """,
        unsafe_allow_html=True
    )

    add_bg_from_local('background app2a.jpg')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
   

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        st.write("")

        if st.button("Classify", key="classify_btn"):
            pred, probs = model_predict(image, model, transform)
            st.session_state.session_state['pred'] = pred
            st.session_state.session_state['probs'] = probs.item()
            st.session_state.session_state['language_selected'] = False

    if st.session_state.session_state['pred'] is not None:
        st.markdown(f"<p style='color: red;'>Prediction: {st.session_state.session_state['pred']}</p>",
                    unsafe_allow_html=True)
        st.markdown(f"<p style='color: red;'>Probability: {st.session_state.session_state['probs']}</p>",
                    unsafe_allow_html=True)

    if st.session_state.session_state['pred'] is not None:
        selected_language = st.selectbox("Select Language", ['English', 'Malayalam'], index=0, key="language_select")
        st.session_state.session_state['selected_language'] = selected_language

    if st.session_state.session_state['pred'] is not None:
        if st.session_state.session_state['selected_language'] == 'Malayalam':
            display_remedies_malayalam(st.session_state.session_state['pred'])
        else:
            display_remedies(st.session_state.session_state['pred'])


if __name__ == "__main__":
    main()
