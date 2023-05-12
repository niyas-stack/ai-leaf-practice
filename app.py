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
         'Remedy for Cassava Mosaic', 'കാസവ മോസായികയുടെ പരിഹാരം'
    ],
    'The above leaf is Cassava CB (Cassava Bacterial Blight)': [
       'Remedy for Cassava Bacterial Blight', 'കാസവ ബാക്ടീരിയൽ ബ്ലൈറ്റിന്റെ പരിഹാരം'
    ]
    # add remedies for other diseases in both English and Malayalam
}

# Create or get the selected_language value from Streamlit's session state
if 'selected_language' not in st.session_state:
    st.session_state['selected_language'] = 'English'
selected_language = st.session_state['selected_language']

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

@st.cache(allow_output_mutation=True)
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


def get_remedy(pred, selected_language):
    remedy = remedies.get(pred)
    if remedy:
        if selected_language == 'English':
            return remedy[0]
        else:
            return remedy[1]
    else:
        return "Remedy not available"


def main():
    global selected_language  # Make selected_language global
    st.set_page_config(page_title="AI Leaf Disease Detection", page_icon=":leaves:")
    st.markdown("<h1 style='color: green;'>AI Leaf Disease Detection</h1>", unsafe_allow_html=True)
    add_bg_from_local('background.jpg')

    # Check if session state already exists, otherwise initialize it
    if not hasattr(session_state, 'selected_language'):
        session_state.selected_language = 'English'

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        st.write("")
        if st.button("Classify", key="classify_btn"):
            pred, probs = model_predict(image, model, transform)
            st.markdown(f"<p style='color: red;'>Prediction: {pred}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: red;'>Probability: {probs.item()}</p>", unsafe_allow_html=True)

            # Language selection for remedy
            selected_language = st.selectbox("Select Remedy Language", ['English', 'Malayalam'], index=0)
            st.session_state['selected_language'] = selected_language

            remedy = get_remedy(pred, selected_language)
            st.markdown("<p style= 'color:red;'>Remedy:</p>", unsafe_allow_html=True)
            st.info(f" {remedy}")


if __name__ == "__main__":
    main()
