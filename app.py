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
remedies = {
    'The above leaf is Cassava (Cassava Mosaic)': 'Remedy for Cassava Mosaic',
    'The above leaf is Cassava CB (Cassava Bacterial Blight)': 'Remedy for Cassava Bacterial Blight'
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

def display_remedies(pred):
    remedy = remedies.get(pred)
    if remedy:
        st.write("remedy:")
        st.info(f" {remedy}")
      

def main():
   st.set_page_config(page_title="AI Leaf Disease Detection", page_icon=":leaves:")
   uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
   if uploaded_file is not None:
       image = Image.open(uploaded_file)
       st.image(image, caption='Uploaded Image', width=300)
       st.write("")
       if st.button("Classify", key="classify_btn"):
           pred, probs = model_predict(image, model, transform)
           st.write(f"Prediction: {pred}")
           st.write(f"Probability: {probs.item()}")
           display_remedies(pred)
          
   # add CSS to set background image
   page_bg_img = '''
   <style>
   body {
   background-image: url();
   background-size: cover;
   }
   </style>
   '''

   st.markdown(page_bg_img, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

