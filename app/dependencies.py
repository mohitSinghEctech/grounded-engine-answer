from fastapi import Request

def get_gemini_client(request: Request):
    return request.app.state.gemini_client