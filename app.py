import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np

from .linear_model import LinearRegressionScratch, LogisticRegressionScratch
from .neural_network import NeuralNetworkScratch, check_gradients_numerical
from .svm import SVMScratch
from .clustering import KMeansScratch, GMMScratch
from .pca import PCAScratch
from .benchmarks import run_scikit_learn_validation_benchmarks

app = FastAPI(title="ML Algorithms Library, Fully From Scratch in NumPy", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

class GradientCheckRequest(BaseModel):
    n_samples: int = 20
    hidden_dim: int = 16
    learning_rate: float = 0.05

class PCARequest(BaseModel):
    n_components: int = 2

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/benchmarks")
async def api_benchmarks():
    bm = run_scikit_learn_validation_benchmarks()
    return {
        "validation_benchmarks": bm
    }

@app.post("/api/gradient_check")
async def api_gradient_check(req: GradientCheckRequest):
    np.random.seed(42)
    X = np.random.randn(req.n_samples, 8)
    y_raw = np.random.randint(0, 3, size=req.n_samples)
    y_onehot = np.zeros((req.n_samples, 3))
    y_onehot[np.arange(req.n_samples), y_raw] = 1.0

    nn = NeuralNetworkScratch(layer_sizes=[8, req.hidden_dim, 3], activation="relu", output_activation="softmax", learning_rate=req.learning_rate)
    rel_diff = check_gradients_numerical(nn, X, y_onehot, eps=1e-5)

    return {
        "relative_gradient_diff": round(float(rel_diff), 8),
        "passed": bool(rel_diff < 1e-4),
        "status": "PASS: Backprop gradients match finite differences!" if rel_diff < 1e-4 else "FAIL"
    }

@app.post("/api/pca_reconstruct")
async def api_pca_reconstruct(req: PCARequest):
    np.random.seed(42)
    X = np.random.randn(100, 10) # 10 dimensions
    pca = PCAScratch(n_components=req.n_components).fit(X)

    rec_err = pca.reconstruction_error(X)
    explained_var_pct = float(np.sum(pca.explained_variance_ratio_) * 100.0)

    return {
        "n_components": req.n_components,
        "explained_variance_percent": round(explained_var_pct, 2),
        "reconstruction_error_mse": round(rec_err, 4)
    }
