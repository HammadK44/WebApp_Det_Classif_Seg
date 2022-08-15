from pydantic import BaseModel


class Todo(BaseModel):
    name: str
    description: str

# This dbmodel.py file is to create the schema, that our table, what fields does it need exactly