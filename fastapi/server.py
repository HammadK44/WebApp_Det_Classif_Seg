import base64
import os
import uuid
import io
from io import BytesIO
import numpy as np
from PIL import Image

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from starlette.responses import Response
from segmentation import load_model, get_segments
from classification import read_image, preprocess, predicti, load_model2
from detection import process_image, annotate_image, load_model3
#from batchclass import predict



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



@app.post("/classification")
async def predict_image(file: bytes = File(...)):
    image = read_image(file)
    image = preprocess(image)
    predictions = predicti(image)
    #byte_io = io.BytesIO()
    #image.save(byte_io, format='PNG')
    return {"Object Name": predictions}
    #json_compatible_item_data = jsonable_encoder(predictions)
    #return JSONResponse(content=json_compatible_item_data)



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
    #byte_io = io.BytesIO()
    #processed_image.save(byte_io, format='PNG')
    #return Response(byte_io.getvalue(), media_type='image/png')




#IMAGE_SIZE = [224, 224]

#train_path = './dataset/train'
#test_path = './dataset/test'

#@app.post("/batchclassif")
#def predictbatch():
#    accuracy, loss, precision, recall, f1score = predict(IMAGE_SIZE, train_path, test_path)
#    return accuracy, loss, precision, recall, f1score 
    