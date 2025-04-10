import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
from skimage.feature import hog, local_binary_pattern
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import joblib
import os
import streamlit as st
 
def load_model(path):
    return joblib.load(path)

# Load models
knn = load_model('/home/pragati/Downloads/project-folder/models/knn_model.pkl')
log_reg = load_model('/home/pragati/Downloads/project-folder/models/logistic_regression_model.pkl')
# rf = load_model('models/random_forest_model.pkl')
# svm = load_model('models/svm_model.pkl')
nb = load_model('/home/pragati/Downloads/project-folder/models/naive_bayes_model.pkl')


class_labels = np.load("/home/pragati/Downloads/project-folder/labels.npy")
 # Fill with your actual class names

# ImageNet transform for ResNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ResNet feature extractor
resnet = models.resnet34(pretrained=True)
resnet.eval()
resnet_feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# --- Preprocessing ---
def denoise_image(image):
    return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

def enhance_contrast(image):
    return cv2.equalizeHist(image)

def apply_filter(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# --- Feature Extraction ---
def extract_features(image):
    # Preprocess
    denoised = denoise_image(image)
    enhanced = enhance_contrast(denoised)
    filtered = apply_filter(enhanced)
    img = cv2.resize(filtered, (128, 128))
    norm_img = img / 255.0

    # HOG
    hog_feat = hog(norm_img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys')

    # LBP
    lbp = local_binary_pattern((norm_img * 255).astype(np.uint8), 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 8+3), range=(0, 8+2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # CNN
    from PIL import Image
    # Convert NumPy image to 3-channel PIL Image
    pil_img = Image.fromarray((norm_img * 255).astype(np.uint8)).convert("RGB")
    img_tensor = transform(np.array(pil_img)).unsqueeze(0)
    with torch.no_grad():
        cnn_feat = resnet_feature_extractor(img_tensor).view(-1).numpy()

    # Combine
    return np.concatenate([hog_feat, lbp_hist, cnn_feat])

# --- Streamlit UI ---
st.title("🖼️ Sketch Classifier")

uploaded_file = st.file_uploader("Upload a sketch image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    img_array = np.array(img)
    st.image(img, caption="Uploaded Sketch", use_column_width=True)

    # Feature extraction
    features = extract_features(img_array).reshape(1, -1)

    # Model selection
  #  model_name = st.selectbox("Choose a model", ["KNN", "Logistic Regression", "Random Forest", "SVM", "Naive Bayes"])
    model_name = st.selectbox("Choose a model", ["KNN", "Logistic Regression", "Naive Bayes"])

    if st.button("Predict"):
        model = {
            "KNN": knn,
            "Logistic Regression": log_reg,
            # "Random Forest": rf,
            # "SVM": svm,
            "Naive Bayes": nb
        }[model_name]

        prediction = model.predict(features)[0]
        st.success(f"🎯 Predicted Label: **{class_labels[prediction]}**")
