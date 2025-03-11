# PGD Adversarial Attack on Facial Recognition 🚨👀

## Overview 🌐
This project showcases a **Projected Gradient Descent (PGD) adversarial attack** on facial recognition systems, using the **FaceNet model**. The goal is to demonstrate how adversarial perturbations can manipulate facial recognition algorithms to make them fail, which is critical for understanding model vulnerabilities in high-security environments like airports. ✈️🔒

### Key Features:
- **PGD Attack** 🔍: Generates adversarial perturbations to fool facial recognition systems.
- **Real-time Visualization** 📊: Use **Streamlit** to interactively upload images, visualize results, and compare distances between embeddings.
- **Adversarial Image Generation** 🖼️: View how facial recognition can be deceived by small image modifications.

## The Scenario 🎭

Imagine a high-security airport where facial recognition systems are used to verify identities. An attacker is able to get access or understand what model airport security is using for facial recognition, and acess to the camera feed. The attacker uses an image of the person and a target image which they want to convert into an adversarial image.By applying the **PGD adversarial attack**, the attacker subtly alters the target image, and inserts it into the camera feed confusing the system into thinking they are not on the ban list and thus bypassing security. 🚨👮‍♂️

### How it Works 🛠️:
1. **Upload Two Faces** 📸: You upload images of two faces—one representing the one on the ban list and the other the current person trying to circumnavigate the security system.
2. **PGD Attack** 💥: The model applies a **PGD attack** to subtly modify the attacker's image.
3. **Face Matching** 🔍: The system calculates embeddings for both faces before and after the attack.
4. **Adversarial Result** ⚠️: The attacker’s image is transformed in a way that increasing the distance between the embeddings, making the system believe the faces are not a match.

### How to Use 📝:
1. Clone this repository and run with streamlit:
   ```bash
   git clone https://github.com/Sslithercode/pgd-adversarial-attack.git
   streamlit run main.py
