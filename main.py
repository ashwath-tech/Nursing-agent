from fastapi import FastAPI, File, UploadFile
from model import NanoVectorDB
from dotenv import load_dotenv
import os
from collections import deque
from openai import OpenAI
import pymupdf

load_dotenv(override=True)
 
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

app = FastAPI()
gemini = OpenAI(api_key=gemini_api_key, base_url= gemini_url)
db = NanoVectorDB()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    file_content = await file.read()
    all_text = ""
    with pymupdf.open(stream=file_content, filetype="pdf") as doc:
        for page in doc:
            all_text += page.get_text().strip() + "\n"
    db.setup(all_text)
    return {"filename": file.filename}

system_message = {"role":"system","content":"You are a helpful assistant. Use the following pieces of retrieved context to answer the user's question. If the answer isn't in the context, say you don't know.DOnt make the answer very long"}
history = deque([],maxlen=10)

@app.post("/question/")
async def send_question(user_question: str):
    if len(history) > 0:
        rewrite_prompt = f"""
        Given this chat history: {list(history)}
        Rewrite this follow-up question to be a standalone search query: '{user_question}'
        Just return the rewritten query, nothing else. Do not answer the question.
        """
        rewrite_res = gemini.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": rewrite_prompt}]
        )
        search_query = rewrite_res.choices[0].message.content.strip()
    else:
        search_query = user_question

    context = db.question(search_query)
    message = [
        system_message,
    ]
    message.extend(history)
    message.append(
        {"role":"user", "content":f''' Here is the context and the question
        Context: {"\n".join(context)}
        Question: {search_query}'''}
    )
    response = gemini.chat.completions.create(model="gemini-2.5-flash", messages=message)
    
    history.append({"role":"user","content":user_question})
    history.append({"role":"assistant","content":response.choices[0].message.content})
    return {"response": response.choices[0].message.content} 