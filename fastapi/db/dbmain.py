from fastapi import FastAPI
import db
import dbmodel


app = FastAPI()


@app.get('/')
def root():
    return {"message":"hello"}


@app.get("/all")
def get_all():
    data = db.all()
    return {"data": data}


@app.post("/create")
def create(data: dbmodel.Todo):
    id = db.create(data)
    return {"inserted": True, "inserted_id": id}


@app.get("/get")            #This is for get one. You can pass the name, and you will get the description
def get_one(name:str):       #Maybe you can use this by giving name as "path" and getting the path location
    data = db.get_one(name) #AND, you can give name of image, and see if present, by seeing if it returns a
    return {"data": data}   #description or not (Description can be the prediction of the image from before)


@app.delete('/delete')
def delete(name:str):
    data = db.delete(name)
    return {"deleted": True, "deleted_count": data}                 
                       


#@app.put("/update")
#def update(name:str, data:models.Todo):
#    data = models.Todo
                        

    
       
                            
                        
