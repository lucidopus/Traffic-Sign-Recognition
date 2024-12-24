import os

import uvicorn
from keras.api.models import load_model
from fastapi import FastAPI, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import APIKeyHeader
from fastapi.requests import Request
from starlette.exceptions import HTTPException

from utils.config import API_KEY, API_NAME, MODEL_PATH
from utils.enums import HttpStatusCode
from utils.pipelines import predict_class_from_uploaded_image
from fastapi import File, UploadFile

model = load_model(MODEL_PATH)

app = FastAPI(
    title=API_NAME,
    description="API for traffic sign detection.",
    version="0.0.1",
    openapi_tags=[
        {
            "name": API_NAME,
            "description": "API for traffic sign detection.",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/robots.txt", include_in_schema=False)
async def get_robots_txt():
    robots_txt_path = os.path.join("static", "robots.txt")
    return FileResponse(robots_txt_path, media_type="text/plain")


templates = Jinja2Templates(directory="static")


@app.get("/", tags=["Index"], response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def get_api_key(
    api_key_header: str = Security(api_key_header),
):
    if api_key_header:
        if api_key_header == API_KEY:
            return API_KEY
        else:
            raise HTTPException(
                status_code=HttpStatusCode.UNAUTHORIZED.value,
                detail="Invalid API Key",
            )
    else:
        raise HTTPException(
            status_code=HttpStatusCode.BAD_REQUEST.value,
            detail="Please enter an API key",
        )


@app.post(
    path="/model/predict",
    tags=["Model Endpoints"],
    response_model=str,
    response_description="Successful Response",
    description="Endpoint to predict traffic sign from an input image.",
)
async def process(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    prediction = predict_class_from_uploaded_image(model=model, uploaded_image=file)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, workers=4)
