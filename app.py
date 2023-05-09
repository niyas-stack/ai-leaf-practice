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
    'The above leaf is Cassava (Cassava Mosaic)': {
        'english': 'Remedy for Cassava Mosaic',
        'malayalam': 'കാസവ മോസായികയുടെ പരിഹാരം'
    },
    'The above leaf is Cassava CB (Cassava Bacterial Blight)': {
        'english': 'Remedy for Cassava Bacterial Blight',
        'malayalam': 'കാസവ ബാക്ടീരിയൽ ബ്ലൈറ്റിന്റെ പരിഹാരം'
    },
    'The above leaf is Cassava Healthy leaf': {
        'english': 'No Remedy Needed for Healthy Cassava Leaf',
        'malayalam': 'ആരോഗ്യമായ കാസവ ഇല പരിഹാരം ആവശ്യമില്ല'
    },
    'The above leaf is Tomato Bacterial spot': {
        'english': 'Remedy for Tomato Bacterial Spot',
        'malayalam': 'ടൊമേറ്റോ ബാക്ടീരിയൽ സ്പോട്ട് പരിഹാരം'
    },
    'The above leaf is Tomato early blight': {
        'english': 'Remedy for Tomato Early Blight',
        'malayalam': 'ടൊമേറ്റോ എർളി ബ്ലൈറ്റിന്റെ പരിഹാരം'
    },
    'The above leaf is Tomato Late blight': {
        'english': 'Remedy for Tomato Late Blight',
        'malayalam': 'ടൊമേറ്റോ ലേറ്റ് ബ്ലൈറ്റിന്റെ പരിഹാരം'
    },
    'The above leaf is Tomato Leaf Mold': {
        'english': 'Remedy for Tomato Leaf Mold',
        'malayalam': 'ടൊമേറ്റോ ലീഫ് മോൾഡ് പരിഹാരം'
    },
    'The above leaf is Tomato Septoria leaf spot': {
        'english': 'Remedy for Tomato Septoria Leaf Spot',
        'malayalam': 'ടൊമേറ്റോ സെപ്റ്റോറിയ ലീഫ് സ്പോട്ട് പരിഹ'
    },
    }
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model_path = "epoch-90.pt"
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
    st.set_page_config(page_title="AI Leaf Disease Detection", page_icon=":leaves:")
    st.markdown("<style>h1{font-family: Arial, sans-serif;}</style>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: green;'>AI Leaf Disease Detection</h1>", unsafe_allow_html=True)
    add_bg_from_local('background.jpg')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        st.write("")
        if st.button("Classify", key="classify_btn"):
            pred, probs = model_predict(image, model, transform)
            st.markdown(f"<p style='color: red;'>Prediction: {pred}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: red;'>Probability: {probs.item()}</p>", unsafe_allow_html=True)
            # Create a dropdown widget to select the language
            language = st.radio("Select remedy's prefered language",('english', 'malayalam'))


             # Get the selected class and display the corresponding remedy in the selected language
            selected_class = classes[pred]
            if selected_class in remedies:
                 remedy = remedies[selected_class][language.lower()]
                 st.write(remedy)
            else:
                 st.write('No remedy found for this class.')



if __name__ == "__main__":
   main()
