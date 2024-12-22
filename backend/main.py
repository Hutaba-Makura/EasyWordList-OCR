from fastapi import FastAPI, HTTPException, UploadFile, File
from models.ocr import ocr_document
import models.wordlist as WL
import tempfile
from fastapi.responses import JSONResponse

app = FastAPI()


# OCR
# ファイルをアップロードしてOCR処理を行う
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile):
    try:
        # ocr_documentを実行
        result = ocr_document(file.file)
        return {"status": "success", "data": result}
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )

# Wordlist
# 単語一覧を取得する
@app.get("/wordlist/")
async def wordlist():
    return {"wordlist": WL.getWords()}

# 単語を追加する
@app.post("/wordlist/add/")
async def add_word(wordset: dict):
    WL.addWord(wordset)

# 単語を編集する
@app.post("/wordlist/edit/")
async def edit_word(wordset: dict, row: int):
    WL.editWord(wordset, row)

# 単語を削除する
@app.post("/wordlist/delete/")
async def delete_word(row: int):
    WL.deleteWord(row)

# 単語一覧を一新する
@app.post("/wordlist/update/")
async def update_wordlist(wordlist: dict):
    WL.updateWordlist(wordlist)