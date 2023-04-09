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
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model_path = "epoch-81.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
summary(model,input_size=(3,224,224))
model.eval()

#pre processing
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
     transforms.RandomAffine(degrees=(25)),
     transforms.RandomRotation(25),
     transforms.RandomHorizontalFlip(0.5),
     transforms.RandomVerticalFlip(0.5),
     transforms.Normalize((0.5,0.5,0.5),(1,1,1))
])

def model_predict(image, model_func, transform):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    output = model_func(image_tensor)
    index = torch.argmax(output)
    pred = classes[index.item()]
    probs, _ = torch.max(F.softmax(output, dim=1), 1)
    if probs < 0.93:
        return "not defined",probs
    else:
        return pred, probs




def main():
  st.set_page_config(
    page_title="AI Leaf Disease Detection",
    page_icon=":leaves:",
    layout="wide",
    initial_sidebar_state="expanded",
)

background = """
<style>
body {
background-image: url("https://cdn.pixabay.com/photo/2017/03/26/14/57/hexagon-2172469_960_720.png");
background-size: cover;
}
</style>
"""

st.markdown(background, unsafe_allow_html=True)

# Add more styling to the button
button_style = """
<style>
    .stFileUploader {
        background-color: #6EBBFF;
        color: white;
        border-radius: 50px;
        padding: 10px 20px;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    .stFileUploader:hover {
        background-color: #4D8FFF;
    }
</style>
"""

st.markdown(button_style, unsafe_allow_html=True)

    st.title("AI Leaf Disease Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=500)
        st.write("")
        st.write("Classifying...")

        # Add CSS styling to loading message
        with st.spinner('Wait for it...'):
            pred, probs = model_predict(image, model, transform)

        st.write(f"Prediction: {pred}")
        st.write(f"Probability: {probs.item()}")

if __name__ == "__main__":
    main()
