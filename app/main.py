"""
FastAPI Application Entry Point.

Mounts Gradio UI and API routers.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import upload, process, status, download

# Create FastAPI app
app = FastAPI(
    title="Video Dubbing Vietnamese",
    description="Hệ thống tự động lồng tiếng Việt cho video",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(process.router, prefix="/api", tags=["Process"])
app.include_router(status.router, prefix="/api", tags=["Status"])
app.include_router(download.router, prefix="/api", tags=["Download"])


@app.get("/")
async def root():
    """Root endpoint — redirect to Gradio UI or return API info."""
    return {
        "service": "Video Dubbing Vietnamese",
        "version": "0.1.0",
        "docs": "/docs",
        "ui": "/ui",
    }


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    from src.m4_pipeline.logger import setup_logger
    setup_logger(log_level="INFO")


# Mount Gradio (import here to avoid circular deps)
def mount_gradio():
    """Mount Gradio interface at /ui."""
    try:
        import gradio as gr
        from app.gradio_ui import create_ui

        gradio_app = create_ui()
        app = gr.mount_gradio_app(app, gradio_app, path="/ui")
    except ImportError:
        pass  # Gradio not installed, skip UI


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
