from fastapi import APIRouter, Depends

from backend.api.dependencies import get_runtime_app
from backend.application.runtime_app import RuntimeApplication

router = APIRouter(prefix="/api/runtime", tags=["Runtime"])


@router.get("/capabilities")
def get_capabilities(
    app: RuntimeApplication = Depends(get_runtime_app),
) -> dict:
    return app.get_capabilities()
