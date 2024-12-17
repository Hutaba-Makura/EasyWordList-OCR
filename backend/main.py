from fastapi import FastAPI, HTTPException, UploadFile, File
from models.ocr import OCR
from models.wordlist import WL

app = FastAPI()


# OCR
@app.post("/ocr/")
async def ocr(file: UploadFile = File(...)):
    ocr = OCR.ocr_document()
    return ocr.process(file)