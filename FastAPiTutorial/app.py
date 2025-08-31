from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Tea(BaseModel):
    id : int
    name : str
    origin : str

teas : List[Tea] = []

@app.get("/")
def home():
    return {"msg":"home page"}


@app.get("/getAllTeas")
def getAllTeas():
    return teas

@app.post("/addTea")
def addTea(tea:Tea):
    teas.append(tea)
    return {"msg" : "Tea is added succesfully"}

@app.put("/updateTea/{tea_id}")
def updateTea(tea_id:int,updated_tea:Tea):
    for i in range(0,len(teas)):
        if teas[i].id == tea_id:
            teas[i] = updated_tea
            return {"msg" : "Tea is updated succesfully"}
        
    return {"msg" : "Tea Not found"}


@app.delete("/deleteTea/{tea_id}")
def deleteTea(tea_id:int):
    for i in range(0,len(teas)):
        if teas[i].id == tea_id:
            teas.pop(i)
            return {"msg" : "Tea is deleted succesfully"}
        
    return {"msg" : "Tea Not found"}
    



