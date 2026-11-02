from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import chat, engines, model_stats

app = FastAPI(title="Predictive Maintenance API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(engines.router)
app.include_router(model_stats.router)
app.include_router(chat.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


class NoCacheStaticFiles(StaticFiles):
    """This app is actively being edited — an editor's browser silently
    serving a stale cached copy of index.html/style.css/app.js after every
    change is far more confusing than the (negligible, localhost-only) cost
    of disabling caching entirely."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


# Vanilla HTML/CSS/JS dashboard, served from the same server as the API —
# no Node.js/build step involved. Registered last so it never shadows /api/*.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/", NoCacheStaticFiles(directory=STATIC_DIR, html=True), name="static")
