import os
import numpy as np
import torch
import PIL
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from werkzeug.utils import secure_filename
import streamlit as st
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 16)
model.to(device)

# Print the model summary
summary(model, (3, 224, 224))

checkpoint_path = "epoch-81.pt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Get the state dict keys for the model and checkpoint
model_state_dict_keys = set(model.state_dict().keys())
checkpoint_keys = set(checkpoint['model_state_dict'].keys())

# Check if all the keys in the checkpoint are present in the model's state dict
missing_keys = model_state_dict_keys - checkpoint_keys
unexpected_keys = checkpoint_keys - model_state_dict_keys
if len(missing_keys) > 0:
    print(f"Missing keys in model's state dict: {missing_keys}")
if len(unexpected_keys) > 0:
    print(f"Unexpected keys in checkpoint: {unexpected_keys}")

# Load the checkpoint
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

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

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def model_predict(image, model_func, transform):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    output = model_func(image_tensor.to(device))
    index = torch.argmax(output)
    pred = classes[index.item()]
    probs, _ = torch.max(F.softmax(output, dim=1), 1)
    return pred, probs


def main():
    st.title("Plant Disease Detection App")
    st.text("Upload a picture of a plant leaf and get a prediction")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and preprocess image
        image = Image.open(uploaded_file)
        image = transform(image)

        # Make prediction
        prediction = model_predict(image=image, model_func=model, transform=transform)

        # Show result
        st.image(image.permute(1, 2, 0), caption=prediction, use_column_width=True)

if __name__ == "__main__":
    main()

