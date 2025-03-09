import streamlit as st
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path

# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Image transformation function
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# PGD Attack Function
def pgd_attack(model, image, epsilon=0.03, alpha=0.005, iters=40):
    perturbed_image = image.clone().detach().requires_grad_(True)
    for _ in range(iters):
        model.zero_grad()  # Zero out gradients before each backward pass
        # Forward pass
        embedding = model(perturbed_image)
        # Loss (you can adjust this as needed for your attack, using the right objective)
        loss = torch.norm(embedding)
        # Backward pass
        loss.backward()
        with torch.no_grad():
            # Apply perturbation to the image based on the gradients
            perturbation = alpha * perturbed_image.grad.sign()
            perturbed_image = perturbed_image + perturbation
            # Clamping to ensure the perturbation is within epsilon bounds
            perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
            perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure it's within image bounds
        # Re-enable gradients for the next iteration
        perturbed_image.requires_grad_(True)
    return perturbed_image.detach()

# Streamlit UI
st.title("🛬 Airport Face Recognition Security Analysis System 🛬")
st.markdown("""
### **The High-Stakes Airport Heist**
A skilled hacker had been following the advancements in face recognition technology, particularly in high-security areas like airports. The authorities had invested heavily in FaceNet to speed up security checks, but the hacker knew that these systems had a **critical flaw**: they could be **fooled** using **adversarial attacks**.
The goal was clear: prove how vulnerable these systems were.
Meanwhile, an unsuspecting traveler arrived at the airport for their flight. Little did they know, their face would become the **target** of an adversarial attack.
""")

# Step 1: Upload Face Images (Travelers' Faces)
st.header("📸 Step 1: Security Checkpoint - Face Capture")
col1, col2 = st.columns(2)
with col1:
    # Add webcam capture option for Face 1 (Traveler 1)
    enable_webcam1 = st.checkbox("Enable Webcam for Face 1 (Traveler 1)")
    if enable_webcam1:
        picture1 = st.camera_input("Capture Face 1", key="face1_cam")
        face1 = picture1 if picture1 else None
    else:
        face1 = st.file_uploader("Upload Face 1 (Traveler 1)", type=["jpg", "png", "jpeg"])
with col2:
    # Add webcam capture option for Face 2 (Random face for attack)
    enable_webcam2 = st.checkbox("Enable Webcam for Face 2 (Random Face)")
    if enable_webcam2:
        picture2 = st.camera_input("Capture Face 2", key="face2_cam")
        face2 = picture2 if picture2 else None
    else:
        face2 = st.file_uploader("Upload Face 2 (Random Face)", type=["jpg", "png", "jpeg"])

if face1 and face2:
    # Convert uploaded images
    image1 = transform(Image.open(face1)).unsqueeze(0)
    image2 = transform(Image.open(face2)).unsqueeze(0)
    
    # Create a visual comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(Image.open(face1))
    axes[0].set_title("Traveler's Face")
    axes[0].axis('off')
    axes[1].imshow(Image.open(face2))
    axes[1].set_title("Target Face")
    axes[1].axis('off')
    st.pyplot(fig)

    # Step 2: Model Testing Interface
    st.header("🔍 Step 2: Security System Testing")
    st.markdown("""
    Before attempting any adversarial attacks, let's test the system's normal functionality.
    """)
    
    # Test the model's recognition capabilities
    st.write("Testing face recognition system...")
    embedding1 = model(image1).detach().numpy()
    embedding2 = model(image2).detach().numpy()
    original_distance = np.linalg.norm(embedding1 - embedding2)
    
    # Display test results
    st.write(f"Face similarity score: {original_distance:.4f}")
    if original_distance < 0.8:
        st.success("Faces recognized as similar!")
    else:
        st.success("Faces recognized as different!")
    
    # Step 3: Attack Parameters
    st.header("⚙️ Step 3: Attack Configuration")
    st.markdown("""
    Configure the attack parameters to test the system's vulnerability.
    """)
    
    col3, col4 = st.columns(2)
    with col3:
        epsilon = st.slider("Perturbation strength (ε)", 0.0, 0.1, 0.03, 0.001)
    with col4:
        iterations = st.slider("Number of iterations", 1, 100, 40)
    
    # Step 4: Execute Attack
    if st.button("Execute Attack"):
        # Execute attack
        st.write("Running PGD attack on Face 1...")
        adv_image1 = pgd_attack(model, image1, epsilon=epsilon, iters=iterations)
        embedding_adv = model(adv_image1).detach().numpy()
        adversarial_distance = np.linalg.norm(embedding_adv - embedding2)
        
        # Save adversarial image
        save_image(adv_image1, "adversarial_face.jpg")
        
        # Show results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(Image.open(face1))
        axes[0].set_title("Original Face")
        axes[0].axis('off')
        axes[1].imshow(Image.open("adversarial_face.jpg"))
        axes[1].set_title("Adversarial Face")
        axes[1].axis('off')
        axes[2].imshow(Image.open(face2))
        axes[2].set_title("Target Face")
        axes[2].axis('off')
        st.pyplot(fig)
        
        # Analyze attack success
        if abs(adversarial_distance - original_distance) > 0.3:
            st.error("The system has been fooled! The faces are no longer close to each other.")
        else:
            st.success("The attack failed. The system identified the faces correctly.")
        
        # Show comparison of distances
        fig, ax = plt.subplots(figsize=(10, 5))
        distances = [original_distance, adversarial_distance]
        labels = ['Before Attack', 'After Attack']
        ax.bar(labels, distances, color=['blue', 'red'])
        ax.set_title('Face Embedding Distance Comparison')
        ax.set_ylabel('Distance')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Step 5: Security Analysis
    st.header("📊 Step 5: Security Analysis")
    st.markdown("""
    This demonstration highlights several critical security implications for face recognition systems in high-security environments like airports:
    """)
    
    # Security Implications
    st.subheader("🔒 Security Implications")
    st.markdown("""
    • **Identity Theft Risk**: Successful attacks can enable unauthorized access to secure areas 
• **System Compromise**: Can lead to false positives or negatives in security checks 
• **Data Breach Potential**: May enable access to sensitive information 
• **Financial Impact**: Can result in significant losses from security breaches 
• **Reputational Damage**: Compromises trust in security systems 

    """)
    
    # Real-World Impact
    st.subheader("🌐 Real-World Impact")
    st.markdown("""
    • **Transportation Security**: Can affect airport security systems 
• **Healthcare Systems**: May impact medical record security 
• **Financial Systems**: Can compromise banking and financial services 
• **Public Safety**: Affects surveillance and law enforcement systems 

    """)
    
    # Mitigation Strategies
    st.subheader("🛡️ Mitigation Strategies")
    st.markdown("""
    • **Adversarial Training**: Training models with attack examples 
• **Gradient Masking**: Protecting against gradient-based attacks 
• **Defensive Distillation**: Reducing model sensitivity to perturbations 
• **Ensemble Methods**: Using multiple models for verification 
• **Input Transformation**: Preprocessing inputs to remove perturbations 
• **Model Hardening**: Strengthening model architecture 
• **Real-time Monitoring**: Continuous system surveillance 

    """)