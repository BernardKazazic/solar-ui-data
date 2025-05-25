from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator
from typing import List, Any
import asyncpg
import os
import logging
import json
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/solar"
)


db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    try:
        # Initialize connection pool
        db_pool = await asyncpg.create_pool(
            DATABASE_URL, min_size=1, max_size=10, command_timeout=60
        )
        print("Database connection pool created successfully!")
    except Exception:
        print("Database connection pool creation failed")

    yield

    # Cleanup: Close the connection pool
    if db_pool:
        await db_pool.close()
        print("Database connection pool closed")


app = FastAPI(title="Solar UI Data API", version="1.0.0", lifespan=lifespan)


class ModelResponse(BaseModel):
    id: int
    name: str
    type: str
    version: int
    features: Any
    plant_name: str

    @field_validator("features")
    @classmethod
    def parse_features(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v


class ModelsListResponse(BaseModel):
    models: List[ModelResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int


@app.get("/")
async def root():
    return {"message": "Solar UI Data API"}


@app.get("/models", response_model=ModelsListResponse)
async def list_models(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
):
    """
    Get a paginated list of all models from the model_metadata table.
    Results are sorted by plant name, then by model name.
    """
    offset = (page - 1) * page_size

    async with db_pool.acquire() as conn:
        try:
            # Query to get total count
            count_query = """
                SELECT COUNT(*) 
                FROM model_metadata mm
                JOIN power_plant_v2 pp ON mm.plant_id = pp.id
            """
            total_count = await conn.fetchval(count_query)

            query = """
                SELECT 
                    mm.id,
                    mm.name,
                    mm.type,
                    mm.version,
                    mm.features,
                    pp.name as plant_name
                FROM model_metadata mm
                JOIN power_plant_v2 pp ON mm.plant_id = pp.id
                ORDER BY pp.name, mm.name
                LIMIT $1 OFFSET $2
            """

            rows = await conn.fetch(query, page_size, offset)

            models = [
                ModelResponse(
                    id=row["id"],
                    name=row["name"],
                    type=row["type"],
                    version=row["version"],
                    features=row["features"],
                    plant_name=row["plant_name"],
                )
                for row in rows
            ]

            total_pages = (total_count + page_size - 1) // page_size

            return ModelsListResponse(
                models=models,
                total_count=total_count,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
            )

        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch models")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
