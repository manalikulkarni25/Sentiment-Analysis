# src/__init__.py

# Import key functions to make them directly accessible from the 'src' package
from .data_preprocessing import clean_text, is_meaningful  # <-- ADDED 'is_meaningful' HERE
from .sentiment_model import analyze_sentiment
from .reporting import generate_report

# The dot (.) before the module name means "from the same package".