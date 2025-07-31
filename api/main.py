"""
Main FastAPI application for the Housing Market Econometrics API
"""

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
import logging
from contextlib import asynccontextmanager

from .core.config import settings, get_cors_config
from .core.dependencies import init_database, cleanup_connections, get_metrics
from .core.security import get_security_headers
from .routers import auth, housing, models, forecasting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Housing Market Econometrics API...")
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Housing Market Econometrics API...")
    try:
        cleanup_connections()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    Housing Market Econometrics API
    
    A comprehensive API for analyzing housing market dynamics and monetary policy transmission
    using advanced econometric models including GJR-GARCH, Regime Switching, Jump Diffusion,
    and Transfer Function models.
    
    ## Features
    
    * **Econometric Models**: GJR-GARCH, Regime Switching, Jump Diffusion, Transfer Function
    * **Data Management**: Housing market data and Federal Reserve rate data
    * **Forecasting**: Multi-horizon forecasts with confidence intervals
    * **Authentication**: JWT-based authentication and API key management
    * **Monitoring**: Prometheus metrics and system health checks
    
    ## Authentication
    
    Most endpoints require authentication. Use the `/auth/login` endpoint to obtain a JWT token,
    then include it in the Authorization header: `Bearer <token>`
    """,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    lifespan=lifespan
)


# Add middleware
app.add_middleware(
    CORSMiddleware,
    **get_cors_config()
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Request/Response middleware for metrics and logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header and log requests"""
    start_time = time.time()
    
    # Get metrics
    metrics = get_metrics()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Add headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers.update(get_security_headers())
    
    # Update metrics
    if metrics:
        metrics.http_requests_total.labels(
            method=request.method,
            path=request.url.path,
            status=response.status_code
        ).inc()
        
        metrics.http_request_duration_seconds.labels(
            method=request.method,
            path=request.url.path
        ).observe(process_time)
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
    )
    
    return response


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.error(f"Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "path": request.url.path
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        from .core.dependencies import get_db
        db = next(get_db())
        db.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.app_version,
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Housing Market Econometrics API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


# API info endpoint
@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Housing Market Econometrics API",
        "endpoints": {
            "auth": "/api/v1/auth",
            "housing": "/api/v1/housing",
            "models": "/api/v1/models",
            "forecasting": "/api/v1/forecasting"
        }
    }


# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(housing.router, prefix="/api/v1/housing", tags=["housing"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(forecasting.router, prefix="/api/v1/forecasting", tags=["forecasting"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
