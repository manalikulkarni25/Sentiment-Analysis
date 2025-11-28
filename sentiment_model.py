# src/sentiment_model.py

# We use 'optimum.pipelines' for ONNX acceleration instead of 'transformers'
from optimum.pipelines import pipeline
from transformers import AutoTokenizer

print("Loading OPTIMIZED sentiment model for maximum CPU speed...")
print("(This may download/convert the model on the first run)...")

# --- START OF MODIFICATION FOR SPEED ---

# 1. Define the base model we want to use (the faster DistilBERT)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# 2. Load the associated tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Create the optimized pipeline using Optimum and the ONNX Runtime (ORT)
sentiment_pipeline = pipeline(
    # --- THIS IS THE CORRECTED LINE ---
    task="text-classification",  # Use the formal task name required by Optimum
    # --- END OF CORRECTION ---
    model=model_name,
    tokenizer=tokenizer,
    accelerator="ort"  # This specifies using the ONNX Runtime
)

# --- END OF MODIFICATION FOR SPEED ---

print("Optimized model loaded successfully.")


def analyze_sentiment(texts):
    """
    Analyzes the sentiment of a list of texts using a pre-trained and OPTIMIZED Transformer model.
    This version includes the human-logic override for known sarcastic phrases.
    """
    # These are our human-logic rules to catch specific failure cases
    SARCASTIC_STARTSWITH = [
        "do whatever",
        "do what ever",
        "sure, whatever",
        "yeah, right"
    ]
    SARCASTIC_CONTAINS_SHORT = [
        "fantastic",
        "hilarious",
        "improve",
        "funny"
    ]

    if isinstance(texts, str):
        texts = [texts]

    # Run the optimized pipeline on all texts for an initial prediction
    model_results = sentiment_pipeline(texts, truncation=True)
    
    # Process the results and apply our human logic overrides
    final_results = []
    for i, result in enumerate(model_results):
        original_text = texts[i].lower().strip()
        
        is_sarcastic = False
        
        # Check 1: Does the text START WITH a known sarcastic phrase?
        for phrase in SARCASTIC_STARTSWITH:
            if original_text.startswith(phrase):
                is_sarcastic = True
                break
        
        if not is_sarcastic:
            # Check 2: Is the text SHORT and CONTAINS a word often used sarcastically?
            if len(original_text.split()) < 4:
                for phrase in SARCASTIC_CONTAINS_SHORT:
                    if phrase in original_text:
                        is_sarcastic = True
                        break

        if is_sarcastic:
            # Override the model's prediction completely
            final_results.append({
                'sentiment': 'Negative',
                'polarity_score': -0.95
            })
            continue # Move to the next text

        # If no override, process the model's original output
        # DistilBERT uses 'POSITIVE' and 'NEGATIVE' labels
        label = result['label']
        score = result['score']
        
        if label == 'NEGATIVE':
            polarity_score = -score
        else: # POSITIVE
            polarity_score = score
        
        final_results.append({
            'sentiment': label.capitalize(),
            'polarity_score': polarity_score
        })
        
    return final_results