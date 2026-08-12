import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to your saved fine-tuned model
save_dir = "./distilbert_plant_model"  # replace with your actual saved model folder

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model.to(device)
model.eval()

# Since you swapped text and label, use model.config.id2label which should match your "text" column
# Normally id2label is filled during training with the "labels", which in your case is actually text
def predict(description):
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = outputs.logits.argmax(dim=-1).item()
    return model.config.id2label[pred_id]  # this now returns the "text" column value (the actual class)

if __name__ == "__main__":
    # Example description
    description = "Leaves show brown circular spots with yellow halo."
    predicted_class = predict(description)
    print("Predicted Disease:", predicted_class)