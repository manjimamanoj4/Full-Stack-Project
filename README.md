# Full-Stack-Project

Plant Disease Diagnosis via Image Upload and Chatbot

Overview

This project is an AI-based system designed to identify plant diseases using two approaches: image upload and text-based input.
It analyzes plant conditions and returns the predicted disease name.



Features

- Image-based plant disease detection
- Text-based disease prediction via chatbot
- Integration of CNN and NLP models
- FastAPI backend for handling requests



Tech Stack

- Frontend: React (Vite), TypeScript
- Backend: FastAPI, Python
- Machine Learning:
  - CNN for image classification
  - DistilBERT for text processing



Project Structure
```

├── frontend/
│   ├── src/
│   └── public/
│
├── backend/
│   ├── main.py
│   ├── predictionn.py
│   └── textinference.py
│
├── notebooks/
│   ├── note.ipynb
│   ├── notebook.ipynb
│   ├── notebook2.ipynb
│   └── notebook(3).ipynb
│
└── README.md

```

How It Works

Image-Based Diagnosis

- User uploads a plant image
- CNN model processes the image
- System predicts the disease

Text-Based Diagnosis

- User describes symptoms
- DistilBERT model analyzes the input
- System returns the predicted disease



Setup Instructions

Backend Setup

cd backend
pip install -r requirements.txt
uvicorn main:app --reload

Frontend Setup

cd frontend
npm install
npm run dev



 
