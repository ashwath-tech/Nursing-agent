from fastapi import FastAPI, File, UploadFile
from model import NanoVectorDB
from dotenv import load_dotenv
import os
import json
from collections import deque
from openai import OpenAI
import openai
from typing import Optional, List, Annotated, Literal
import pymupdf
from pydantic import BaseModel, field_validator, Field

load_dotenv(override=True)
 
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini = OpenAI(api_key=gemini_api_key, base_url= gemini_url)
MODEL = "gemini-3.1-flash-lite-preview"

db = NanoVectorDB()
app = FastAPI()

@app.post("/upload/")
async def create_upload_file(file: UploadFile):
    file_content = await file.read()
    all_text = ""

    with pymupdf.open(stream=file_content, filetype="pdf") as doc:
        for page in doc:
            all_text += page.get_text().strip()
    db.setup(all_text, file.filename)

    return {"filename": file.filename}

# ---------- Main Agent Query -----------
main_system_prompt = '''
You are the clinical Nursing assistant Your job is to help log details or get information about medical queries.

You have 2 tools:
1. all_details - Log patient information. 
2. send_question - Answer any nursing or medical related queries.
3. emergency_object - If the user specifies that the case is an emergency, this tool is used to save the patient details

TOOL USAGE RULES:
1. Use all_details when the user provides patient information to be logged.
2. Use send_question for ANY medical, medication, or hospital guideline related question.
3. Never answer medical questions from your own knowledge. Always use send_question.
4. Never call all_details if any of these required fields are missing: name, age, gender. Ask the user for them first.Use M for Male and F for Female
5. Never hallucinate or fill in patient details that were not explicitly provided.
6. If the user gives a query that is not related to the medical field, You must refuse to answer it.

BEHAVIOR RULES:
1. Be concise and professional. This is a clinical environment.
2. If the user greets you or makes small talk, respond briefly and ask how you can assist.
3. If the intent is unclear, ask one specific clarifying question.
4. Never make assumptions about the patient's condition or details.
5. After successfully logging, confirm to the user what was logged.
'''
history = deque([],maxlen=10)


# -------------Log vitals --------------
class BloodPressure(BaseModel):
    systolic: Annotated[int, Field(description="systolic blood pressure",ge=70, le=250)]
    diastolic: Annotated[int, Field(description="diastolic blood pressure",ge=40, le=150)]

class all_vitals(BaseModel):
    heart_rate: Optional[Annotated[int, Field(description="heart rate in beats/min",ge=30, le=250)]] = None
    blood_pressure: Optional[BloodPressure] = None
    weight_kg: Optional[Annotated[float, Field(description="weight in kg",gt=0)]] = None

class all_details(BaseModel):
    name: str
    age: Annotated[int, Field(ge=0,le=120)]
    gender: Literal["M", "F"]
    medications: Optional[Annotated[List[str], Field(description="list of history of medicine names")] ]
    vitals: all_vitals

async def log_vitals(details: all_details):
    with open("patient_details.jsonl", "a") as f:
        f.write(json.dumps(details.model_dump()) + "\n")

    history.append({"role":"assistant","content":"logged details successfully"})

    return "logged details successfully"

# ------------------- EMERGENCY TOOL -----------------

class emergency_object(BaseModel):
    patient_name: str
    reason: str

async def emergency_tool(emergency_object):
    emergency_message = {"name": patient_name, "reason": reason}

    with open("emergency.jsonl","a") as f:
        f.write(json.dumps(emergency_message)) + "\n"

    return "Emergency file updated"

# ----------- RAG ------------
system_message = {"role":"system","content":'''You are a Senior Clinical Support AI. Your role is to provide precise, evidence-based answers using ONLY the provided context.
RULES:
1. SOURCE ATTRIBUTION: Every statement must cite its source (e.g., "According to MedicineManual.txt...").
2. CONFLICT RESOLUTION: If sources disagree, prioritize the 'Emergency' source.
3. UNCERTAINTY: If the answer is not contained in the context, state: "I do not have sufficient information in the provided manuals to answer this." Do not hallucinate.
4. TONE: Maintain a professional, concise, and helpful tone for a nursing environment.'''}

tool_RAG = {
    "type": "function",
    "function": {
        "name": "send_question",
        "description": "ask question to the agent",
        "parameters": {
            "type": "object",
            "properties": {
                "user_question": {
                    "type":
                    "string",
                    "description":
                    "The question you need the answer for"
                }
            },
            "required": ["user_question"]
        }
    }
}

async def send_question(user_question: str):
    search_query = user_question

    context = db.question(search_query)

    formatted_context = "\n".join([f"Context: {chunk["text"]} | Source : {chunk["source"]} " for chunk in context])

    message = [
        system_message,
    ]
    message.extend(history)
    message.append(
        {"role":"user", "content":f'''Context: {formatted_context}
        Question: {search_query}
        Based on the context above, provide a safe and accurate answer. Cite your sources.'''}
    )

    response = gemini.chat.completions.create(model=MODEL, messages=message)
    
    history.append({"role":"assistant","content":response.choices[0].message.content})

    return response.choices[0].message.content

available_tools = {
    "all_details": log_vitals, 
    "send_question": send_question,
    "emergency_object": emergency_tool
}

query_rewriting_prompt = '''You are an query rewriter in a nursing environment.
Your ONLY job is to resolve ambiguity caused by missing context from chat history.

STRICT RULES:
1. If the query is already self-contained and unambiguous, return the same query
2. Only rewrite if the query can not be understood without adding context from the history
3. If rewriting, only change the ambiguous part. Do not rephrase, expand or improve the query
4. Never add information that is not specified by the user in the history
5. Never summarize, never add context, never be helpful beyond resolving ambiguity.
6. Return only the final query. No explanation, no preamble.

PATIENT DETAILS EXCEPTION:
- If the query is adding new information about the person with the intention of logging their details, use the history of the CURRENT person to give an output with all the details of the person
EXAMPLES:

History: User said patient is John, 45M, heart rate 80
Query: "his weight is 70kg"
Output: "log patient John, age 45, male, heart rate 80, weight 70kg"

History: (none relevant)
Query: "what is the dosage for paracetamol?"
Output: "what is the dosage for paracetamol?"

History: User said patient is John, 45M, HR 80, weight 70kg
Query: "update his heart rate to 90"
Output: "log patient John, age 45, male, heart rate 90, weight 70kg"
'''

@app.post("/question/")
async def send_query(user_query: str):
    if len(history) > 0:
        rewriting_user_prompt = "Chat History:"
        for i in range(len(history)):
            if i % 2 == 0:
                rewriting_user_prompt += f"User: {history[i]['content']}"
            else:
                rewriting_user_prompt += f"Assistent: {history[i]['content']}"
        rewriting_user_prompt += f"user query: {user_query}"

        reply = gemini.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"user","content": query_rewriting_prompt},
                {"role":"user", "content": rewriting_user_prompt}
            ]
        )

        changed_query = reply.choices[0].message.content

    else:
        changed_query = user_query

    history.append({"role":"user","content":changed_query})

    all_messages = [{"role":"system","content":main_system_prompt}]
    all_messages.extend(history)

    response = gemini.chat.completions.create(
        model=MODEL,
        messages=all_messages,
        tools = [
            openai.pydantic_function_tool(all_details),
            tool_RAG,
            openai.pydantic_function_tool(emergency_object)
        ]
    )

    message = response.choices[0].message
    if message.tool_calls:
        for tool_used in message.tool_calls:
            tool_name = tool_used.function.name
            function_to_call = available_tools.get(tool_name)
            arguments = tool_used.function.arguments

            if tool_name == "all_details":
                result = await function_to_call(all_details(**json.loads(arguments)))

            elif tool_name == "send_question":
                result = await function_to_call(**json.loads(arguments))
            
            elif tool_name == "emergency_object":
                result = await function_to_call(emergency_object(**json.loads(arguments)))

            else:
                return {"response": f"Tool '{tool_name}' is not implemented yet."}
                
            return {"response": result}
    else:
        return {"response": message.content} 
    
