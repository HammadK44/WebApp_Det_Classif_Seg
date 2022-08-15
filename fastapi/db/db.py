import pymongo


mongoURL = "mongodb://localhost:27017/TODO"    # TODO here is the database name to be created
client = pymongo.MongoClient(mongoURL)


db = client["TODO"]
collection = db["todo"]      # Now the database can be accessed using this Collection named "collection"
                             # You can see inside the database, collection name as "todo"     
collection_Single = db["todo1"]
collection_Batch = db["todo2"]


def create(data):
    data = dict(data)
    response = collection.insert_one(data)     # this function is returning the inserted_id
    return str(response.inserted_id)       # inserted_id is the unique id of a record in the collection
                                                # Mongodb will always create this id automatically
    
    # IDs in Mongodb are "object IDs". ObjectIDs cannot be returned in JSON format
    # So the ID in the "return response.inserted_id" should be converted to string format 






def all():
    response = collection.find({})      # will "find" return all and any of the data inside the collection
    #response = collection.find({"_id":0})
    #response = collection.find({"_id":0, "name": 1})  #If you make something 0, it will not return it.
                        # If you make something 0 and make another one 1, it will return ONLY that 1 one. 
    
    data = []
    for i in response:
        i["_id"] = str(i["_id"])   #Again, Ids cannot be returned in json. converting all data ids to str.
        data.append(i)
    return data


def get_one(condition):
    response = collection.find_one({"name": condition})     #condition should be passed as a filter
                                                            # which should always be a dictionary
    response["_id"] = str(response["_id"])
    return response                                         


def update(name, data):
    response = collection.update_one({"name": name, "$set": data})
    return response.modified_count


def delete(name):
    response = collection.delete_one({"name": name})
    return response.deleted_count



