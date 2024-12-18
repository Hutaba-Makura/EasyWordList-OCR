from fastapi import FastAPI, HTTPException, UploadFile, File
from models.ocr import ocr_document
import models.wordlist as WL
import tempfile

app = FastAPI()


# OCR
# ファイルをアップロードしてOCR処理を行う
@app.post("/ocr/")
async def ocr(file: UploadFile = File(...)):
    # ファイルを一時保存して処理
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    # OCR処理
    coords_list, merged_boxes = ocr_document(temp_file_path)

    return {"coords_list": coords_list, "merged_boxes": merged_boxes}

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