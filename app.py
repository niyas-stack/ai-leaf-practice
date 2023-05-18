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


def model_predict(image, model_func, transform):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    output = model_func(image_tensor)
    index = torch.argmax(output)
    pred = classes[index.item()]
    probs, indices = torch.topk(F.softmax(output, dim=1), 3)
    pred_prob = [(classes[indices[i].item()], probs[0][i].item() * 100) for i in range(3)]
    return pred, pred_prob


def run_app():
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
                unsafe_allow_html=True)

    st.markdown("""
    <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
      <a class="navbar-brand" href="https://youtube.com/dataprofessor" target="_blank">Data Professor</a>
      <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav">
          <li class="nav-item active">
            <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="https://youtube.com/dataprofessor" target="_blank">YouTube</a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">Twitter</a>
          </li>
        </ul>
      </div>
    </nav>
    """, unsafe_allow_html=True)

    st.title("Leaf Disease Classification")

    st.set_option('deprecation.showfileUploaderEncoding', False)

    def load_image(image_file):
        img = Image.open(image_file)
        return img

    def process_image(image):
        img = image.convert('RGB')
        return img

    def predict(image):
        pred, pred_prob = model_predict(image, model, transform)
        return pred, pred_prob

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "Run the app"])

    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Run the app":
        st.sidebar.success('To change the image, browse again from the top.')

        st.info("1. Upload an image of a leaf.")
        st.info("2. The app will predict the disease on the leaf.")
        st.info("3. The app will provide remedies to the predicted disease.")
        st.info("4. Select the language (English/Malayalam) for the remedies.")

        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = load_image(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            image = process_image(image)

            if st.button("Predict"):
                pred, pred_prob = predict(image)
                st.success(f"Prediction: {pred}")
                st.info(f"Confidence: {pred_prob[0][1]:.2f}%")
                st.info(f"Top 3 Predictions:")
                for i, (pred, prob) in enumerate(pred_prob):
                    st.write(f"{i + 1}. {pred}: {prob:.2f}%")

                if pred in remedies:
                    remedy_info = remedies[pred]
                    st.info("Remedies:")
                    st.write(f"{remedy_info[0]}")
                    if selected_language == 'English':
                        st.audio(remedy_info[2], format='audio/m4a')
                    elif selected_language == 'Malayalam':
                        st.audio(remedy_info[3], format='audio/m4a')

        st.sidebar.title("Language")
        selected_language = st.sidebar.selectbox("Select the language for the remedies",
                                                 ["English", "Malayalam"])

        st.sidebar.title("About")
        st.sidebar.info(
            """
            This app is a simple leaf disease classification application.
            The model is trained to predict diseases on leaves from three different plant types: cassava, tomato, and bean.
            """
        )


if __name__ == '__main__':
    run_app()

