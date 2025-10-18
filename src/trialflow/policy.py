import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY = os.getenv("API_KEY", "").strip()
DISABLE_AUTH = os.getenv("DISABLE_AUTH", "0") == "1"

x_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
authz = APIKeyHeader(name="Authorization", auto_error=False)

async def require_api_key(x_key: str = Security(x_api_key), auth: str = Security(authz)):
    if DISABLE_AUTH or not API_KEY:
        return None
    if auth and auth.startswith("Bearer ") and auth.split(" ", 1)[1] == API_KEY:
        return API_KEY
    if x_key and x_key == API_KEY:
        return API_KEY
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
