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
classes = dict({0:'The above leaf is Cassava (Cassava Mosaic) ', 
                1:'The above leaf is Cassava CB (Cassava Bacterial Blight)', 
                2:'The above leaf is Cassava Healthy leaf', 
                3:'This is not trained yet',
                4:'The above leaf is Tomato Bacterial spot', 
                5:'The above leaf is Tomato early blight',
                6:'The above leaf is Tomato Late blight',
                7:'The above leaf is Tomato Leaf Mold', 
                8:'The above leaf is Tomato Septoria leaf spot',
                9:'The above leaf is Tomato Spider mites Two-spotted spider mite', 
                10:'The above leaf is Tomato Target Spot',
                11:'The above leaf is Tomato Yellow Leaf Curl Virus', 
                12:'The above leaf is Tomato mosaic virus', 
                13:' The above leaf is Tomato healthy', 
                14:'The above leaf is bean angular leaf spot',
                15:'The above leaf is bean healthy', 
                16:'The above leaf is bean rust'})

remedies = {
    'The above leaf is Cassava (Cassava Mosaic) ': [
         'Use of resistant variety Sripadmanaba suited for Tamil Nadu and Kerala. Mosaic tolerant varieties such as H-97 may be used to minimize economic loss of tubers. Select setts from healthy plants. Roug out and destroy infected plants in the field at early stage.Control whitefly by installing yellow sticky traps, removal of weed hosts, spray neem oil (20 ml / litre of water). Spray Dimethoate 30 EC (2 ml / litre of water) to control the vector.', 'കിഴങ്ങുവർഗ്ഗങ്ങളുടെ സാമ്പത്തിക നഷ്ടം കുറയ്ക്കുന്നതിന് H-97 പോലുള്ള മൊസൈക്ക് സഹിഷ്ണുതയുള്ള ഇനങ്ങൾ ഉപയോഗിക്കാം. ആരോഗ്യമുള്ള ചെടികളിൽ നിന്ന് സെറ്റുകൾ തിരഞ്ഞെടുക്കുക. രോഗബാധയക്ക് ആദ്യത്തെ സ്റ്റേജിൽ കിട്ടുന്ന അമൃതം വെള്ളിയിൽ കൊണ്ടുവരുന്നതായിരിക്കാം, അതിനാൽ വെട്ടിച്ചെടികൾ നന്നായിരിക്കണം. വെള്ളിയെക്കുറിച്ചുള്ള പിഴച്ചെടികൾ നീങ്ങിനിർത്തുക, വിറക് ഓയില് (20 മി.എല് കൂട്ടിച്ചേർക്കുക) സ്പ്രേ ചെയ്യുക. വെട്ടിച്ചെടികൾ നന്നായിരിക്കണം. വൈറ്റ്ഫ്ലൈയെ നിയന്ത്രിക്കുന്നതിനും വളർത്തിപ്പാട്ടും വീഴ്ചയും വേർതിരിക്കുക.',
         'CASSAVA(MOSAIC)(ENG).mp3', 'CASSAVA(MOSAIC)(MAL).m4a'
    ],
    'The above leaf is Cassava CB (Cassava Bacterial Blight)': [
       'Chemical control is not effective. Use of resistant or tolerant varieties is the only method of control. Among improved varieties, H-97, H-226, H-1687 and H-2304 are tolerant to the disease while H-165 is highly susceptible. Among the local varieties, M-4 is tolerant to the disease.', 'രാസ നിയന്ത്രണം ഫലപ്രദമല്ല. പ്രതിരോധശേഷിയുള്ളതോ സഹിഷ്ണുതയുള്ളതോ ആയ ഇനങ്ങളുടെ ഉപയോഗം മാത്രമാണ് നിയന്ത്രണ മാർഗ്ഗം. മെച്ചപ്പെട്ട ഇനങ്ങളിൽ, H-97, H-226, H-1687, H-2304 എന്നിവ രോഗത്തെ സഹിഷ്ണുതയുള്ളവയാണ്, അതേസമയം H-165 വളരെ രോഗസാധ്യതയുള്ളവയാണ്. പ്രാദേശിക ഇനങ്ങളിൽ, M-4 രോഗത്തോട് സഹിഷ്ണുത പുലർത്തുന്നു',
       'cassava bacterial blight eng.mp3', 'cassava bacterialblight mal.m4a'
    ],
     'The above leaf is Tomato Bacterial spot': [
       'Hot water treatment of seeds at 50°C for 25 minutes is effective. Crop rotation with non-host crop helps in reducing the disease incidence. Soaking of seed in solution of Agrimycin-100 (1 gram / 10 litre of water) or Streptocycline (0.5 gram/litre of water) or with Pseudomonas fluorescens (20 gram/litre of water / kg of seed) for 30 minute protects the seedlings in the initial stages of growth.Soil application of Pseudomonas fluorescens fortified in neem cake also reduces the disease incidence. Affected plants should be removed and destroyed. Disinfect the area with bleaching powder @ 15 kg / ha in endemic fields.The planting pits should be prophylatically drenched with Copper oxychloride 50 WP (2 gram/litre of water) or Copper hydroxide 77 WP (1 gram/litre of water) two weeks before planting. Soil drench Agrimycin-100 (1 gram / 10 litre of water) or Streptocycline (0.5 gram/litre of water) when the disease is noticed.', 'വിത്ത് 50 ഡിഗ്രി സെൽഷ്യസിൽ 25 മിനിറ്റ് ചൂടുവെള്ളത്തിൽ ശുദ്ധീകരിക്കുന്നത് ഫലപ്രദമാണ്. ആതിഥേയമല്ലാത്ത വിളകൾ ഉപയോഗിച്ച് വിള ഭ്രമണം ചെയ്യുന്നത് രോഗബാധ കുറയ്ക്കാൻ സഹായിക്കുന്നു. വിത്ത് അഗ്രിമൈസിൻ-100 (1 ഗ്രാം / 10 ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ സ്ട്രെപ്റ്റോസൈക്ലിൻ (0.5 ഗ്രാം/ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ സ്യൂഡോമോണസ് ഫ്ലൂറസെൻസ് (20 ഗ്രാം/ലിറ്റർ വെള്ളം / കിലോ വിത്ത്) എന്നിവയിൽ 30 മിനിറ്റ് മുക്കിവയ്ക്കുന്നത് തൈകളെ സംരക്ഷിക്കുന്നു. വളർച്ചയുടെ പ്രാരംഭ ഘട്ടത്തിൽ.വേപ്പിൻ പിണ്ണാക്ക് ഉറപ്പിച്ച സ്യൂഡോമോണസ് ഫ്ലൂറസെൻസ് മണ്ണിൽ പ്രയോഗിക്കുന്നതും രോഗബാധ കുറയ്ക്കുന്നു. ബാധിച്ച ചെടികൾ നീക്കം ചെയ്യുകയും നശിപ്പിക്കുകയും വേണം. എൻഡമിക് വയലുകളിൽ ഹെക്ടറിന് 15 കി.ഗ്രാം ബ്ലീച്ചിംഗ് പൗഡർ ഉപയോഗിച്ച് പ്രദേശം അണുവിമുക്തമാക്കുക.നടുന്നതിന് രണ്ടാഴ്ച മുമ്പ്, നടീൽ കുഴികളിൽ കോപ്പർ ഓക്സിക്ലോറൈഡ് 50 WP (2 ഗ്രാം/ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ കോപ്പർ ഹൈഡ്രോക്സൈഡ് 77 WP (1 ഗ്രാം/ലിറ്റർ വെള്ളം) ഉപയോഗിച്ച് നനയ്ക്കണം. രോഗം ശ്രദ്ധയിൽപ്പെട്ടാൽ അഗ്രിമൈസിൻ-100 (1 ഗ്രാം / 10 ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ സ്ട്രെപ്റ്റോസൈക്ലിൻ (0.5 ഗ്രാം/ലിറ്റർ വെള്ളം) മണ്ണിൽ നനയ്ക്കുക.',
       'cassava bacterial blight eng.mp3', 'cassava bacterialblight mal.m4a'
    ]
    # add remedies for other diseases in both English and Malayalam
}

selected_language = 'English'  # Set the default language


num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model_path = "epoch-8.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
def model_predict(image, model_func, transform):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    output = model_func(image_tensor)
    index = torch.argmax(output)
    pred = classes[index.item()]
    probs, _ = torch.max(F.softmax(output, dim=1), 1)
    if probs < 0.93:
        pred = 'This is not trained yet'
        return pred, probs
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
text_area_style = """
    background-color: black;
    color: white;
    font-family: Arial, sans-serif;
"""
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
            st.text_area("", value=remedy[0], height=200)
        else:
            st.text_area("", value=remedy[1], height=200)

def display_remedies_malayalam(pred):
    remedy = remedies.get(pred)
    if remedy:
        st.markdown("<p style='color:white;'>Remedy (Malayalam):</p>", unsafe_allow_html=True)
        audio_file = remedy[3]
        with open(audio_file, 'rb') as audio:
            st.audio(audio.read(), format='audio/mp3')
        st.text_area("", value=remedy[1], height=200)
# Initialize SessionState
def init_session_state():
    if 'session_state' not in st.session_state:
        st.session_state.session_state = {
            'pred': None,
            'probs': None,
            'selected_language': 'English',
            'language_selected': False
        }

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
        if st.session_state.session_state['pred'] != 'This is not trained yet' and st.session_state.session_state['pred'] != 'The above leaf is Cassava Healthy leaf' and st.session_state.session_state['pred'] != ' The above leaf is Tomato healthy' and st.session_state.session_state['pred'] != 'The above leaf is bean healthy':
            selected_language = st.selectbox("Select Language", ['English', 'Malayalam'], index=0, key="language_select")
            st.session_state.session_state['selected_language'] = selected_language

    if st.session_state.session_state['pred'] is not None and not st.session_state.session_state['language_selected']:
        if st.session_state.session_state['selected_language'] == 'Malayalam':
            display_remedies_malayalam(st.session_state.session_state['pred'])
        else:
            display_remedies(st.session_state.session_state['pred'])

def about_page():
    st.title("About Dr.Leaf")
    st.write("Dr.Leaf is a sophisticated and powerful application designed to accurately identify plant diseases based on leaf images. It offers comprehensive remedies for these diseases in both English and Malayalam, catering to a wide range of users. Our primary focus is on individuals actively involved in kitchen gardening, ensuring that they have the necessary tools to maintain the health of their plants.")
    st.write("Dr.Leaf is the result of diligent teamwork by our dedicated team members: Laura, Niyas, Poornima, and Pranav. Their collective efforts have culminated in an application that seamlessly combines advanced technology with practical solutions for plant disease identification and management.")
    st.write("We are committed to providing a user-friendly experience, empowering kitchen gardening enthusiasts to make informed decisions and take appropriate actions to safeguard their plants. Dr.Leaf strives to be a reliable companion for individuals seeking expert guidance in plant disease detection and treatment.")



def contact_page():
    st.title("Contact Dr.Leaf")
    st.write("For any inquiries or support, please contact us at:")
    st.write("- Email ids:")
    st.write("  - laura.saleena@gmail.com")
    st.write("  - niyasmohammad16@gmail.com")
    st.write("  - poornimababus2001@gmail.com")
    st.write("  - pranavrajiv2000@gmail.com")



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
    add_bg_from_local("background app2a.jpg")
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
