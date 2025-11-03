from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.nn.functional as F

# Initialize flask app
app = Flask(__name__, template_folder="templates")
CORS(app)

# Load model and tokenizer
MODEL_PATH = "./my_roberta_model"
LABEL_MAP = {0: "false", 1: "true"}
#print("Loading model and tokenizer...")

def load_model_and_tokenizer(model_path=MODEL_PATH):
    """Load the trained RoBERTa model and tokenizer."""
    print("Loading model and tokenizer...")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print("Model and tokenizer loaded successfully!!")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, "cpu"

def predict_text(model, tokenizer, device, text):
    """Perform inference for a single text input."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()

    prediction = LABEL_MAP.get(pred_id, "unknown")
    return prediction, confidence

# load model before starting
model, tokenizer, device = load_model_and_tokenizer()


## routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not tokenizer:
        return jsonify({'error': 'Model is not loaded properly.'}), 500

    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided.'}), 400

        prediction, confidence = predict_text(model, tokenizer, device, text)

        response = {
            "prediction": prediction,
            "confidence": confidence
        }

        print(f"Text: {text[:50]}... | Prediction: {prediction} ({confidence:.2f})")
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed.'}), 500


## run app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
