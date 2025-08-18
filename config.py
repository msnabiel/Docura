generation_config = {
    "response_mime_type": "application/json",
    "response_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence_score": {"type": "number"},
            "need_api": {
                "type": "object",
                "nullable": True,
                "properties": {
                    "type": {"type": "string", "enum": ["GET", "POST"]},
                    "url": {"type": "string"},
                    "headers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "value": {"type": "string"}
                            },
                            "required": ["key", "value"]
                        }
                    },
                    "body": {"nullable": True}
                },
                "required": ["type", "url"]
            }
        },
        "required": ["answer", "confidence_score", "need_api"]
    }
}

"""
Return ONLY this JSON (no markdown, no extra text):
{{
  "answer": "combine answers into key-word rich, descriptive,string",
  "confidence_score": <float between 0 and 1>,
  "need_api": false | {{
    "type": "GET" | "POST",
    "url": "<api or webpage URL>",
    "headers": <optional>,
    "body": <optional>
  }}
}}"""
