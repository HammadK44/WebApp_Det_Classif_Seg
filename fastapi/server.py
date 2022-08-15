import io
import numpy as np
from PIL import Image
from pydantic import BaseModel
import json
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.responses import Response
from segmentation import load_model, get_segments
from classification import read_image, preprocess, predicti, load_model2
from detection import process_image, annotate_image, load_model3
from batchclass import predict

import sys
from db import db



model1 = load_model()
model2 = load_model2()
model3 = load_model3()

app = FastAPI()



@app.post("/segmentation")
def get_segmentation_map(image: bytes = File(...)):
    segmented_image = get_segments(model1, image)
    bytes_io = io.BytesIO()
    segmented_image.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")


#-------------------------------------------------------------------------------------------------------

@app.post("/classification")
#async def predict_image(file: UploadFile = File(...)):
async def predict_image(file: bytes = File(...)):
    image = read_image(file)
    image = preprocess(image)
    predictions = predicti(image)
    return predictions


#-------------------------------------------------------------------------------------------------------


@app.post("/detection")
async def get_detection(image: bytes = File(...)):
    image = read_image(image)
    image = np.array(image)
    detections, confidence_threshold = process_image(image)
    processed_image = annotate_image(image, detections, confidence_threshold)
    processed_image = Image.fromarray(processed_image)
    byte_io = io.BytesIO()
    processed_image.save(byte_io, format='PNG')
    return Response(byte_io.getvalue(), media_type='image/png')


#-------------------------------------------------------------------------------------------------------


@app.post("/batchclassification")
async def predict_image(request: Request):

    getpath = 'YES'
    x = db.collection_Batch.find_one({"GetPath": getpath})
    x["_id"] = str(x["_id"])
    x = str(x)
    x = x.split("'", 12)
    x = x[-2]

    accuracy, loss, precision, recall, f1score = predict(x)

    record_dict ={
        "Dataset": x,
        "Accuracy": accuracy,
        "Loss": loss,
        "Precision": precision,
        "Recall": recall,
        "F1Score": f1score
        }

    insertInTable= db.collection_Batch.insert_one(record_dict)

    pass



    #return {"Accuracy": accuracy, 
    #        "Loss": loss, 
    #        "Precision": precision, 
    #        "Recall": recall, 
    #        "F1Score": f1score}


    
if __name__ == '__main__':                                  # Remove when Dockerizing
    uvicorn.run(app, host='127.0.0.1', port=8000)
