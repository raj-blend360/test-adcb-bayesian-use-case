import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.database import create_tables
from api.routers import upload, transform, model, results, tune, visualize, optimize

app = FastAPI(title="Bayesian MMM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(transform.router)
app.include_router(model.router)
app.include_router(results.router)
app.include_router(tune.router)
app.include_router(visualize.router)
app.include_router(optimize.router)


@app.on_event("startup")
def on_startup():
    create_tables()


@app.get("/health")
def health():
    return {"status": "ok"}
