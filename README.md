# Phishing Link Detector

## Objective
This project aims to detect phishing links given a URL input. It includes:
- **Phishing Link Detection:** Classifies URLs as legitimate or phishing.
- **Database Integration:** Utilizes a MongoDB cloud database to store input URLs.
- **Machine Learning Model:** Uses a Multi-Layer Perceptron (MLP) model for classification.

## Frontend
The frontend is built for ease of use:
- **HTML/CSS:** Developed with HTML and CSS for a clean, responsive interface.
- **Flask Application:** Deployed using a Flask web application to handle user input and display results.
- **Tableau Dashboard:** Includes an interactive Tableau dashboard to show cyberattack rates across India, particularly in South India.

![WhatsApp Image 2024-03-25 at 07 48 54_2aa35422](https://github.com/2211Jarl/Malc-Inc./assets/75835715/5225144c-b21c-412f-9fe4-0544e9982675)

![WhatsApp Image 2024-03-30 at 10 36 12_57424b92](https://github.com/2211Jarl/Malc-Inc./assets/75835715/766b9fc7-de18-456d-a2f1-f98f4d1d7849)

![WhatsApp Image 2024-03-30 at 10 36 13_ebd55a4f](https://github.com/2211Jarl/Malc-Inc./assets/75835715/23d64ff6-b5d9-4de1-bfbe-8b4c5156f1fe)

## Backend
The backend manages data processing and model predictions:
- **Normalization:** Input data is normalized using the MinMax Scaler.
- **MLP Model:** A Multi-Layer Perceptron (MLP) model is used to classify URLs as legitimate or phishing.

## Getting Started

### Prerequisites
Ensure the following are installed on your system:
- **Python 3.x**
- **Flask**
- **MongoDB** (cloud database)
- **Tableau** (for dashboard integration, license or account may be needed)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/2211Jarl/Malc-Inc/phishing-link-detector.git
   cd phishing-link-detector
