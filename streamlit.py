import streamlit as st
import cv2
import numpy as np
import pickle
from groq import Groq
from PIL import Image
import matplotlib.pyplot as plt  # For pie chart

st.set_page_config(page_title="Damage Detection App", layout="wide")

st.markdown(
    "<h1 style='text-align: center; font-size: 48px; font-weight: bold;'>"
    "AI-Based Goods Damage Detection and Email Reporting Application</h1>",
    unsafe_allow_html=True,
)

# 🔐 API Key
groq_api_key = "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"
client = Groq(api_key=groq_api_key)

# Load model
@st.cache_resource
def load_model():
    with open('goods_damage_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Feature extraction
def extract_features_from_bytes(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (100, 100))
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Email generation
def generate_email(
    damaged,
    non_damaged,
    product_type,
    shipment_method,
    packaging_type,
    fragile_label,
    transit_days,
    template_type,
):
    subject = f"Inspection Results – {product_type} Shipment"

    if template_type == "Internal Team":
        prompt = f"""
Subject: {subject}
Dear Team,

The recent shipment of **{product_type}** has been inspected.

Shipment Details:
- Shipment Method: {shipment_method}
- Packaging Used: {packaging_type}
- Fragile Label Applied: {"Yes" if fragile_label else "No"}
- Time in Transit: {transit_days} days

Inspection Summary:
- Damaged Items: {damaged}
- Non-Damaged Items: {non_damaged}

This is an internal inspection report for documentation purposes.

Best regards,  
Quality Control Team
"""
    elif template_type == "Vendor Notification":
        prompt = f"""
Subject: Damage Found in Shipment – {product_type}
Dear Vendor,

We have completed the inspection of the recent shipment of **{product_type}**.

Details:
- Shipment Method: {shipment_method}
- Packaging Type: {packaging_type}
- Fragile Label: {"Yes" if fragile_label else "No"}
- Transit Time: {transit_days} days

Inspection Results:
- Damaged Packages: {damaged}
- Non-Damaged Packages: {non_damaged}

Please review the shipment and ensure packaging compliance in future dispatches.

Regards,  
Quality Control Department
"""
    elif template_type == "Customer Notification":
        prompt = f"""
Subject: Update on Your Shipment – {product_type}
Dear Customer,

We’re writing to update you on your recent shipment of **{product_type}**.

Shipment Method: {shipment_method}  
Packaging: {packaging_type}  
Fragile Label: {"Yes" if fragile_label else "No"}  
Transit Duration: {transit_days} days  

Inspection Outcome:
- Damaged Units: {damaged}
- Non-Damaged Units: {non_damaged}

If any action is needed on your order, our support team will reach out to you shortly.

Best regards,  
Customer Support  
"""
    else:
        return "⚠️ Invalid email template selected."

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "system", 
                "content": "You are a professional quality control manager writing a formal email."
            }, {
                "role": "user", 
                "content": prompt
            }],
            temperature=0.5,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating email: {e}"

# Session state initialization
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "results" not in st.session_state:
    st.session_state.results = []

if "counts" not in st.session_state:
    st.session_state.counts = {"damaged": 0, "non_damaged": 0}

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["🔧 Inputs & Upload", "🖼 Detection & Results", "📧 Generated Email"])

# ---------- TAB 1 ----------
with tab1:
    st.header("🔧 Enter Shipment Details & Upload Images")

    col1, col2 = st.columns([1, 1])

    with col1:
        product_type = st.selectbox("Product Type", ["Electronics", "Furniture", "Clothing", "Food", "Other"])
        shipment_method = st.selectbox("Shipment Method", ["Air", "Sea", "Land", "Courier"])
        packaging_type = st.selectbox("Packaging Type", ["Bubble Wrap", "Foam", "Cardboard Box", "Wooden Crate", "None"])
        fragile_label = st.checkbox("Fragile Label Applied?")
        transit_days = st.number_input("Transit Time (Days)", min_value=0, step=1)
        template_type = st.selectbox("Email Template", ["Internal Team", "Vendor Notification", "Customer Notification"])

    with col2:
        uploaded_files = st.file_uploader("📥 Upload Product Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"{len(uploaded_files)} image(s) uploaded successfully.")

            st.markdown("### 📸 Uploaded Preview")
            cols = st.columns(4)
            for i, file in enumerate(uploaded_files):
                img = Image.open(file)
                cols[i % 4].image(img, caption=file.name, use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    st.header("🖼 Image Damage Detection")

    if not st.session_state.uploaded_files:
        st.warning("⚠️ No images uploaded. Please go to 'Inputs & Upload' tab first.")
    else:
        if st.button("🚀 Run Damage Detection"):
            results = []
            damaged = 0
            non_damaged = 0

            for file in st.session_state.uploaded_files:
                file.seek(0)
                features = extract_features_from_bytes(file).reshape(1, -1)
                prediction = model.predict(features)[0]
                label = "Damaged" if prediction == 0 else "Not Damaged"
                if label == "Damaged":
                    damaged += 1
                else:
                    non_damaged += 1

                results.append((file.name, Image.open(file), label))

            st.session_state.results = results
            st.session_state.counts = {"damaged": damaged, "non_damaged": non_damaged}
            st.success("✅ Damage detection complete!")

        if st.session_state.results:
            st.subheader("📊 Summary")

            # Side-by-side layout for counts and pie chart
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 📦 Detection Count")
                st.metric("✅ Not Damaged", st.session_state.counts['non_damaged'])
                st.metric("❌ Damaged", st.session_state.counts['damaged'])

            with col2:
                st.markdown("### 🥧 Damage Ratio")
                labels = ['Damaged', 'Not Damaged']
                sizes = [
                    st.session_state.counts["damaged"],
                    st.session_state.counts["non_damaged"]
                ]
                colors = ['#FF4B4B', '#4CAF50']
                explode = (0.05, 0)

                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                wedges, texts, autotexts = ax.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    explode=explode,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops=dict(color="white", fontsize=10)
                )

                for text in texts:
                    text.set_fontsize(9)
                for autotext in autotexts:
                    autotext.set_fontsize(8)

                ax.axis('equal')
                st.pyplot(fig)

            # Show classified images
            st.subheader("🔍 Classification Results")
            cols = st.columns(4)
            for i, (filename, img, label) in enumerate(st.session_state.results):
                caption = f"{filename} - {'❌ Damaged' if label == 'Damaged' else '✅ Not Damaged'}"
                cols[i % 4].image(img, caption=caption, use_container_width=True)

# ---------- TAB 3 ----------
with tab3:
    st.header("📧 Auto-Generated Email")

    if not st.session_state.results:
        st.warning("⚠️ No inspection results found. Please run detection in Tab 2.")
    else:
        if st.button("✉️ Generate Email"):
            email = generate_email(
                st.session_state.counts["damaged"],
                st.session_state.counts["non_damaged"],
                product_type,
                shipment_method,
                packaging_type,
                fragile_label,
                transit_days,
                template_type
            )
            st.session_state.email_text = email
            st.success("📨 Email generated!")

        if st.session_state.email_text:
            st.subheader("📄 Email Content")
            st.code(st.session_state.email_text, language="markdown")
