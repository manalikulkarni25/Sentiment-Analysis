# src/data_preprocessing.py
import re

def clean_text(text):
    """
    Cleans the input text by converting to lowercase, removing special characters,
    and extra whitespace.
    """
    # If the input is not a string (like a missing value), return an empty string.
    if not isinstance(text, str):
        return ""
    
    text = text.lower()  # Convert to lowercase
    # Keep letters, numbers, and basic punctuation that can show emphasis (!, ?)
    text = re.sub(r'[^a-zA-Z0-9\s.!?]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def is_meaningful(text):
    """
    Checks if the text is meaningful enough to be sent to the model.
    Returns False for very short or gibberish-like text.
    """
    # Rule 1: Must not be empty
    if not text:
        return False
        
    # Rule 2: Must be longer than 3 characters, with exceptions for common words
    if len(text) <= 3:
        whitelist = ['good', 'bad', 'great', 'sad', 'ok', 'yes', 'no']
        if text in whitelist:
            return True
        return False
        
    # Rule 3: Must contain at least one vowel (simple check for gibberish like 'rghhh')
    if not re.search(r'[aeiou]', text):
        return False
        
    return True