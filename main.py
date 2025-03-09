import streamlit as st
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

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
st.title("Face Recognition PGD Adversarial Attack Demo")

# Story Introduction: The Heist at the Airport
st.markdown("""
### **The High-Stakes Airport Heist**
A skilled hacker had been following the advancements in face recognition technology, particularly in high-security areas like airports. The authorities had invested heavily in FaceNet to speed up security checks, but the hacker knew that these systems had a **critical flaw**: they could be **fooled** using **adversarial attacks**.

The goal was clear: prove how vulnerable these systems were.

Meanwhile, an unsuspecting traveler arrived at the airport for their flight. Little did they know, their face would become the **target** of an adversarial attack.
""")

# Step 1: Upload Face Images (Travelers' Faces)
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
    st.image([face1, face2], caption=["Original Face 1", "Face 2 (Random Face)"], width=150)
    
    # Story Progression: The Attack Begins
    st.markdown("""
    ### **Step 2: The Attack Begins**
    With the images ready, the hacker runs the **PGD attack**. The goal? Subtly alter Traveler 1's face so that it appears to be the random face, fooling the FaceNet system into thinking it’s a completely different person.
    """)

    # Compute embeddings for original faces
    embedding1 = model(image1).detach().numpy()
    embedding2 = model(image2).detach().numpy()
    original_distance = np.linalg.norm(embedding1 - embedding2)
    
    # Step 3: Apply PGD Adversarial Attack on Face 1 (Traveler 1)
    st.write("Running PGD attack on Face 1...")
    adv_image1 = pgd_attack(model, image1)
    embedding_adv = model(adv_image1).detach().numpy()
    adversarial_distance = np.linalg.norm(embedding_adv - embedding2)

    # Save adversarial image
    save_image(adv_image1, "adversarial_face.jpg")
    
    # Show adversarial image
    st.image("adversarial_face.jpg", caption="Adversarial Face (Modified Traveler 1)", width=150)
    
    # Step 4: Visualize the Attack Impact
    st.markdown("""
    ### **Step 4: The Moment of Truth**
    The system is now tricked. The hacker had successfully altered the face, causing the system to misidentify Traveler 1.
    """)

    # Show comparison of distances before and after attack
    st.write(f"**Distance Before Attack:** {original_distance:.4f}")
    st.write(f"**Distance After Attack:** {adversarial_distance:.4f}")
    
    if abs(adversarial_distance - original_distance) > 0.3:
        st.error("The system has been fooled! The faces are no longer close to each other.")
    else:
        st.success("The attack failed. The system identified the faces correctly.")

    # Step 5: The Reaction
    st.markdown("""
    ### **Step 5: The Reaction**
    Security officers are confused. The traveler is flagged as a mismatch, and the system tells them that they are not the person it expects, forcing them to manually review the identity.
    """)

    # Provide download option for adversarial image
    st.download_button("Download Adversarial Image", 
                      open("adversarial_face.jpg", "rb").read(), 
                      "adversarial_face.jpg")
    
    # Step 6: The Resolution
    st.markdown("""
    ### **Step 6: The Resolution**
    Eventually, the attack is discovered, and the security team investigates. The system is proven to be vulnerable to **adversarial attacks**, raising concerns about its reliability for high-security applications.
    """)

    # Step 7: Conclusion
    st.markdown("""
    ### **Conclusion: The Future of Face Recognition Systems**
    This demonstration highlights the vulnerabilities in face recognition systems, urging the need for more **robust** systems that are resilient to adversarial examples. 
    In the future, these systems may need to train with adversarial examples to become more secure.
    """)

