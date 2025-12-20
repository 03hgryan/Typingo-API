from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

import models
from database import engine
from routers import users

load_dotenv()

# Only creates when tables do not exist
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AST - ASR + translation",
    description="FastAPI for AST",
    version="1.0.0",
)

# Security headers for production only
if os.getenv("ENVIRONMENT") == "production":
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware

    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            return response

    app.add_middleware(HTTPSRedirectMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

# CORS configuration
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "API for ast",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "users": "/api/users",
        },
    }


# Include routers
app.include_router(
    users.router,
    prefix="/api/users",
    tags=["users"],
    responses={404: {"description": "User not found"}},
)


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "FastAPI for AST"}
