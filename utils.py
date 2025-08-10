import re
import json
import requests
from typing import Optional
# ------------------------------- CLEANING FUNCTIONS -------------------------------
def clean_string_post(text: str) -> str:
    """
    Clean a string by:
    - Replacing escaped quotes (\" -> ")
    - Removing unescaped backslashes (\)
    - Replacing newlines with space
    - Replacing em dashes (\u2014) with --
    - Collapsing multiple spaces
    """
    import re

    # Replace escaped double quotes
    cleaned = text.replace('\\"', '"')

    # Replace em dash Unicode with "--"
    cleaned = cleaned.replace('\u2014', '--')

    # Remove unescaped backslashes (not part of escape sequences)
    # Use regex to avoid removing backslashes that are part of Unicode escapes
    cleaned = re.sub(r'\\(?!u[0-9a-fA-F]{4})', '', cleaned)

    # Replace one or more newline characters with a space
    cleaned = re.sub(r'\n+', ' ', cleaned)

    # Collapse multiple spaces into one
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

def clean_text_for_gemini(raw_text: str) -> str:
    """
    Cleans and formats messy input text into a single readable paragraph.
    Designed for LLM input (e.g., Gemini, GPT).
    """

    # Remove excessive spaces and normalize line breaks
    text = raw_text.strip()
    text = re.sub(r'\r\n|\r', '\n', text)             # Normalize newlines
    text = re.sub(r'\n{2,}', '\n', text)              # Collapse multiple blank lines
    text = re.sub(r'[ \t]+', ' ', text)               # Collapse multiple spaces/tabs
    text = re.sub(r'\n', '. ', text)                  # Treat newlines as sentence ends
    text = re.sub(r'\.\s*\.', '.', text)              # Remove accidental double periods
    text = re.sub(r'\s*\.\s*', '. ', text)            # Ensure spacing after periods
    text = re.sub(r'\s{2,}', ' ', text)               # Final whitespace cleanup

    # Capitalize first letter of the paragraph
    text = text[0].upper() + text[1:] if text else ""

    return text.strip()

# === Utility: Clean OCR Text ===
def clean_ocr_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    return text.strip()

# ------------------------------- EXTRACTING FROM JSON FUNCTIONS -------------------------------

def extract_answer_from_json_block(text: str) -> str:
    """
    Extracts the 'answer' field from any embedded JSON block in the text.
    Returns an empty string if not found or if JSON is malformed.
    """
    # Find the first JSON-looking block
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    
    if json_match:
        try:
            json_obj = json.loads(json_match.group())
            return json_obj.get("answer", "").strip()
        except json.JSONDecodeError:
            return ""
    
    return ""

def extract_final_answer(text: str) -> str:
    """
    Extracts the most relevant answer from the input text.
    If a JSON block with an 'answer' field exists, that is returned.
    Otherwise, the full text is returned as-is.
    """
    # Search for JSON object
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    
    if json_match:
        try:
            json_obj = json.loads(json_match.group())
            if "answer" in json_obj:
                return json_obj["answer"].strip()
        except json.JSONDecodeError:
            pass  # Ignore malformed JSON

    # No JSON or no answer field — return full string
    return text.strip()

def extract_json_from_response(response: str) -> dict:
    try:
        # Remove Markdown-style ```json blocks
        cleaned = re.sub(r"```(?:json)?", "", response).strip("` \n")

        # Try loading as JSON
        parsed_json = json.loads(cleaned)
        
        # Ensure required fields exist
        if "answer" not in parsed_json:
            parsed_json["answer"] = ""
        if "confidence_score" not in parsed_json:
            parsed_json["confidence_score"] = 0.5
        if "need_api" not in parsed_json:
            parsed_json["need_api"] = False
            
        return parsed_json

    except Exception:
        # Fallback: handle plain text like "Answer: ..."
        cleaned = cleaned.strip()
        if cleaned.lower().startswith("answer:"):
            return {
                "answer": cleaned[len("answer:"):].strip(),
                "confidence_score": 0.5,
                "need_api": False
            }

        # Otherwise return as plain text under "answer"
        if cleaned:
            return {
                "answer": cleaned,
                "confidence_score": 0.5,
                "need_api": False
            }

        raise ValueError("Failed to parse Gemini response: Empty or invalid format.")
    
# ------------------------------- CALLING EXTERNAL APIs -------------------------------

def call_external_api(need_api: dict) -> str:
    """
    Calls an external API based on the Gemini `need_api` object.
    
    Args:
        need_api (dict): A dict with keys like 'type', 'url', 'headers', and 'body'.
        
    Returns:
        str: API response text (or JSON as a string).
    """
    method = need_api.get("type", "GET").upper()
    url = need_api.get("url")
    headers = need_api.get("headers", {})
    body = need_api.get("body", {})

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=body, timeout=10)
        else:
            return f"Unsupported request method: {method}"

        if response.status_code == 200:
            # Try to return JSON string if possible
            try:
                return response.json()
            except Exception:
                return response.text
        else:
            return f"API returned status code {response.status_code}"

    except Exception as e:
        return f"API call failed: {str(e)}"
    
# ------------------------------- URL CLASSIFICATION -------------------------------
    
def classify_url_by_response(url: str) -> str:
    """
    Classifies a URL based on what the server returns.

    Returns:
        - 'document': if it's a file (PDF, DOCX, XLSX, etc.)
        - 'web': if it's an HTML page
        - 'api': if it's JSON (e.g., API response)
        - 'error': if request fails or response is invalid
        - 'unknown': if type can't be determined
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            return 'error'

        content_type = response.headers.get('Content-Type', '').lower()

        # Common document types
        if any(doc_type in content_type for doc_type in [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument",
            "application/vnd.ms-excel",
            "text/plain",
            "text/csv"
        ]):
            return 'document'

        # JSON = API
        if "application/json" in content_type:
            return 'api'

        # HTML = web page
        if "text/html" in content_type:
            return 'web'

        return 'unknown'
    except Exception:
        return 'error'

# ------------------------------- URL CHECK -------------------------------
def is_url(string: str) -> bool:
    """Check if string is a URL"""
    return string.startswith(('http://', 'https://'))

