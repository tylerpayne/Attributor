from typing import List

from fastapi import HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from loguru import logger

from server.attribution_api import AttributionAPI
from server.models import ConfigParams, Message

# Initialize the application
app = AttributionAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Model-Level Endpoints


@app.get("/api/v1/model/inputs")
async def get_model_inputs():
    return app.attribution.inputs


@app.get("/api/v1/model/flow")
async def get_model_flow():
    return app.attribution.flow


@app.get("/api/v1/model/outputs")
async def get_model_outputs():
    return app.attribution.outputs


# Layer-Level Endpoints


@app.get("/api/v1/model/layers/{layer_index}/inputs")
async def get_layer_inputs(layer_index: int):
    try:
        return app.attribution.layers[layer_index].inputs
    except IndexError:
        raise HTTPException(status_code=404, detail="Layer not found")


@app.get("/api/v1/model/layers/{layer_index}/flow")
async def get_layer_flow(layer_index: int):
    try:
        return app.attribution.layers[layer_index].flow
    except IndexError:
        raise HTTPException(status_code=404, detail="Layer not found")


@app.get("/api/v1/model/layers/{layer_index}/outputs")
async def get_layer_outputs(layer_index: int):
    try:
        return app.attribution.layers[layer_index].outputs
    except IndexError:
        raise HTTPException(status_code=404, detail="Layer not found")


@app.get("/api/v1/model/layers/{layer_index}/attention_head_weights")
async def get_attention_head_weights(layer_index: int):
    try:
        return app.attribution.layers[layer_index].attention_head_weights
    except IndexError:
        raise HTTPException(status_code=404, detail="Layer not found")


@app.get("/api/v1/model/layers/{layer_index}/pre_residual")
async def get_pre_residual(layer_index: int):
    try:
        return app.attribution.layers[layer_index].pre_residual
    except IndexError:
        raise HTTPException(status_code=404, detail="Layer not found")


# Attention Head-Level Endpoints


@app.get("/api/v1/model/layers/{layer_index}/attention_heads/{head_index}/inputs")
async def get_attention_head_inputs(layer_index: int, head_index: int):
    try:
        return app.attribution.layers[layer_index].attention_heads[head_index].inputs
    except IndexError:
        raise HTTPException(status_code=404, detail="Layer or attention head not found")


@app.get("/api/v1/model/layers/{layer_index}/attention_heads/{head_index}/flow")
async def get_attention_head_flow(layer_index: int, head_index: int):
    try:
        return app.attribution.layers[layer_index].attention_heads[head_index].flow
    except IndexError:
        raise HTTPException(status_code=404, detail="Layer or attention head not found")


@app.get("/api/v1/model/layers/{layer_index}/attention_heads/{head_index}/outputs")
async def get_attention_head_outputs(layer_index: int, head_index: int):
    try:
        return app.attribution.layers[layer_index].attention_heads[head_index].outputs
    except IndexError:
        raise HTTPException(status_code=404, detail="Layer or attention head not found")


@app.get(
    "/api/v1/model/layers/{layer_index}/attention_heads/{head_index}/attention_scores"
)
async def get_attention_scores(layer_index: int, head_index: int):
    try:
        return (
            app.attribution.layers[layer_index]
            .attention_heads[head_index]
            .attention_scores
        )
    except IndexError:
        raise HTTPException(status_code=404, detail="Layer or attention head not found")


# POST Endpoint to update the global model


@app.post("/api/v1/attribute")
async def attribute(messages: List[Message]):
    params = app.attribute(messages)
    return params

@app.post("/api/v1/configure")
async def configure(config: ConfigParams):
    try:
        app.configure(config)
    except Exception as ex:
        logger.exception(ex)
        return HTTPException(500, detail=f"{type(ex).__name__}: {ex}")
    
@app.get("/api/v1/configured")
async def configured():
    logger.debug(app.configured)
    return app.configured