from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

# Pull in the load_models function (and underlying `models` dict)
from fast_api_integration.models import load_models

# Import your router
from fast_api_integration.api.solve import router as solve_router

app = FastAPI(
    title="BlockBlast Solver API",
    description="FastAPI endpoints for running RL-based BlockBlast solvers",
    version="1.0.0",
)

# Add CORS middleware to allow everything
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Hook: load all SB3 models once at startup
@app.on_event("startup")
def on_startup():
    load_models()


# Mount the /api/solve router
app.include_router(solve_router, prefix="/api")
