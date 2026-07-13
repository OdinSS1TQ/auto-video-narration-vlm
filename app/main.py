"""
FastAPI Application Entry Point.

Mounts Gradio UI and API routers.
"""

from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.routers import upload, process, status, download
from app.routers import health, config, tts

# Create FastAPI app
app = FastAPI(
    title="Video Dubbing Vietnamese",
    description="Hệ thống tự động lồng tiếng Việt cho video tiếng Anh "
                "sử dụng Vision-Language Model + Zero-shot Voice Cloning.",
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
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(config.router, prefix="/api", tags=["Config"])
app.include_router(tts.router, prefix="/api", tags=["TTS"])

# Mount Studio React Frontend
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if not frontend_dist.exists():
    frontend_dist.mkdir(parents=True, exist_ok=True)
    (frontend_dist / "index.html").write_text(
        "<html><head><title>AI Dub Studio Building</title></head>"
        "<body style='background-color:#0d0d0f;color:#e5e1e4;font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;'>"
        "<div style='text-align:center;border:1px solid #1c1b1d;background-color:#141416;padding:40px;border-radius:8px;max-width:500px;'>"
        "<h1 style='color:#adc6ff;margin-bottom:16px;'>Studio Frontend is building...</h1>"
        "<p style='color:#c2c6d6;font-size:14px;line-height:1.6;'>Vite compiled files were not found. Please run <code>npm run build</code> in the <code>frontend/</code> directory to compile assets.</p>"
        "<p style='color:#707585;font-size:12px;margin-top:24px;'>Checking again in 5 seconds...</p>"
        "</div>"
        "<script>setTimeout(() => location.reload(), 5000);</script>"
        "</body></html>",
        encoding="utf-8"
    )
app.mount("/studio", StaticFiles(directory=str(frontend_dist), html=True), name="studio")


# Global exception handler for unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return structured JSON error."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


@app.get("/")
async def root():
    """Root endpoint — return API info and links."""
    return {
        "service": "Video Dubbing Vietnamese",
        "version": "0.1.0",
        "docs": "/docs",
        "ui": "/ui",
        "studio": "/studio",
        "endpoints": {
            "upload_video": "POST /api/upload/video",
            "upload_audio": "POST /api/upload/audio",
            "process": "POST /api/process",
            "cancel_job": "DELETE /api/process/{job_id}",
            "status": "GET /api/status/{job_id}",
            "list_jobs": "GET /api/jobs",
            "download_video": "GET /api/result/{job_id}",
            "download_srt": "GET /api/result/{job_id}/srt",
            "result_report": "GET /api/result/{job_id}/report",
            "health": "GET /api/health",
            "config": "GET /api/config",
        },
    }


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    from src.m4_pipeline.logger import setup_logger
    setup_logger(log_level="INFO")

    from app.routers.process import load_jobs
    load_jobs()


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
