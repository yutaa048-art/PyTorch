import random

def generate_http_request(i: int) -> str:
    endpoints = ["/login", "/api/users", "/api/data", "/checkout", "/logout", "/chat", "/profile", "/settings"]
    methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    status_codes = ["200 OK", "404 Not Found", "500 Internal Server Error", "401 Unauthorized", "201 Created", "403 Forbidden"]
    
    method = random.choice(methods)
    endpoint = random.choice(endpoints)
    status = random.choice(status_codes)
    
    body = ""
    if method in ["POST", "PUT", "PATCH"]:
        body = f'{{"user_id": {i}, "action": "{method}", "endpoint": "{endpoint}"}}'
        
    return f"""{method} {endpoint} HTTP/1.1
Host: localhost
Authorization: Bearer xxx_token_{i}
Content-Type: application/json

{body}

Response {status}
"""

def generate_http_dataset(count: int = 5000) -> list[str]:
    return [generate_http_request(i) for i in range(count)]
