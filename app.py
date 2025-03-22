from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from models import *
from config import get_db, init_db
from auth_utils import hash_password, verify_password, create_access_token, verify_access_token
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, List
from sqlalchemy.orm import Session
from datetime import timedelta
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
import requests
from config import Base
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import google.generativeai as genai
app = FastAPI()

init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Или укажите конкретные домены, например, ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (GET, POST, DELETE и т.д.)
    allow_headers=["*"],  # Разрешить все заголовки
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
@app.post("/register", response_model=dict)
def register_user(user: UserCreate, db: Session = Depends(get_db)) -> dict:
    db_user = db.query(UserInDB).filter(UserInDB.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    if user.role not in ["doctor", "patient"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    hashed_pw = hash_password(user.password)
    
    if user.role == "doctor":
        new_user = UserInDB(
            name=user.name,
            email=user.email,
            hashed_password=hashed_pw,
            role=user.role
        )
        db.add(new_user)
        
        new_doctor = DoctorsInDB(
            name=user.name,
            email=user.email,
            hashed_password=hashed_pw,
            doctor_type=user.doctor_type,
            experience=user.experience,
            rating=user.rating,
            patient_count=user.patient_count
        )
        db.add(new_doctor)
    else:
        new_user = UserInDB(
            name=user.name,
            email=user.email,
            hashed_password=hashed_pw,
            role=user.role,
            gender=user.gender,
            dateOfBirth=user.dateOfBirth,
            phone=user.phone,
            address=user.address,
            condition=user.condition,
            riskLevel=user.riskLevel,
            bloodType=user.bloodType
        )
        db.add(new_user)

    db.commit()
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role},
        expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer", "role": user.role}


@app.post("/login/", response_model=Token)
def login_for_access_token(user: UserLogin, db: Session = Depends(get_db)) -> Token:
    db_user = db.query(UserInDB).filter(UserInDB.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.email, "role": db_user.role}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "role": db_user.role}

@app.get("/me", response_model=Dict)
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Базовая информация для всех пользователей
    response = {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role
    }
    
    if user.role == "doctor":
        doctor = db.query(DoctorsInDB).filter(DoctorsInDB.email == user_email).first()
        if doctor:
            response.update({
                "doctor_type": doctor.doctor_type,
                "experience": doctor.experience,
                "rating": doctor.rating,
                "patient_count": doctor.patient_count
            })
    else:
        response.update({
            "gender": user.gender,
            "dateOfBirth": user.dateOfBirth,
            "phone": user.phone,
            "address": user.address,
            "condition": user.condition,
            "riskLevel": user.riskLevel,
            "bloodType": user.bloodType,
            "lastVisit": user.lastVisit
        })
    
    return response

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

genai.configure(api_key='AIzaSyBNZ9RJAIcuuLlhCj8KtbxoC6opxY_5q5E')

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
  model_name="tunedModels/untitled-prompt1-6chq2zgt8t9p",
  generation_config=generation_config,
)

def send_request_to_gemini(symptoms: str, api_key: str, db: Session) -> str:
    # Получаем список врачей из базы данных
    doctors = db.query(DoctorsInDB).all()
    doctor_names = ', '.join([doctor.doctor_type for doctor in doctors])

    # Конфиг генерации текста для Gemini
    generation_config = genai.GenerationConfig(
        temperature=0.7,
        max_output_tokens=256
    )

    # Используем generative model (если нужно)
    model = genai.GenerativeModel(
        model_name="tunedModels/untitled-prompt1-6chq2zgt8t9p",
        generation_config=generation_config
    )

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(symptoms)

    # Формируем новый запрос для выбора врача
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
    
    new_prompt = (
        f"CHOOSE A DOCTOR FROM THE LIST BASED ON THE GIVEN SYMPTOMS: {doctor_names}. "
        f"PICK A DOCTOR AND CREATE A VERY SHORT CHAT TITLE (e.g., 'Facial-Irritation') and color for chat pfp. Symptoms: {symptoms}"
        f"OUTPUT EVERYTHING IN THIS FORMAT: doctor short-description rgb(0,0,0)."
    )

    data = {
        "contents": [{"parts": [{"text": new_prompt}]}]
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        best_doctor = requests.post(url, json=data, headers=headers)
        best_doctor.raise_for_status()  
        best_doctor_json = best_doctor.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error from Gemini API: {str(e)}")
    
    if response:
        return best_doctor_json, response.text
    else:
        raise HTTPException(status_code=500, detail="No response from Gemini API")

@app.post("/submit_request/")
async def submit_request(
    symptoms: str = Form(...),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    payload = verify_access_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    
    if not user_email:
        raise HTTPException(status_code=400, detail="User email not found in token")
    
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    
   
    
    gem_response = ""
    gemini_response = ""

    best_doctor1, gemini_response = send_request_to_gemini(
        symptoms=symptoms, 
        api_key="AIzaSyBNZ9RJAIcuuLlhCj8KtbxoC6opxY_5q5E", 
        db=db
    )
    
    gem_response = best_doctor1['candidates'][0]['content']['parts'][0]['text'].strip()

    best_doctor, chat_title, color = gem_response.split()
    doctor = db.query(DoctorsInDB).filter(DoctorsInDB.doctor_type == best_doctor).first()

    if not doctor:
        raise HTTPException(status_code=404, detail=f"Doctor {best_doctor} not found")

    best_doctor_id = doctor.id
    best_doctor_name = doctor.name

    user_request = UserRequest(
        user_id=user.id,
        name = user.name,
        image_path="uploads",
        symptoms=symptoms,
        color=color,
        chat_title=chat_title,
        response=gemini_response,
        doctor_name = best_doctor_name,
        status = True,
        doctor_id=best_doctor_id
    )
    db.add(user_request)
    db.commit()

    return {"msg": "Request submitted successfully", "gemini_response": gemini_response}


@app.get("/my_requests/", response_model=list)
def get_user_requests(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> list:
    payload = verify_access_token(token)
    user_email = payload.get("sub")
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    requests = db.query(UserRequest).filter(UserRequest.user_id == user.id).all()
    return [{"title": req.chat_title.replace("-", " "), "color": req.color, "createdAt": req.createdAt, "symptoms": req.symptoms, "response": req.response} for req in requests]

@app.get("/my_appointments", response_model=List[dict])
async def get_my_appointments(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    current_time = datetime.datetime.now()
    appointments = []
    
    # Если пользователь - доктор
    if user.role == "doctor":
        doctor = db.query(DoctorsInDB).filter(DoctorsInDB.email == user_email).first()
        if not doctor:
            raise HTTPException(status_code=404, detail="Doctor not found")
            
        appointments = db.query(Appointment).filter(
            Appointment.doctor_id == doctor.id
        ).order_by(Appointment.start_time.desc()).all()
        
        return [{
            "id": app.id,
            "start_time": app.start_time,
            "end_time": app.end_time,
            "patient": {
                "id": app.patient_id,
                "name": app.patient_name
            },
            "status": "upcoming" if app.start_time > current_time else "past"
        } for app in appointments]
    
    # Если пользователь - пациент
    else:
        appointments = db.query(Appointment).filter(
            Appointment.patient_id == user.id
        ).order_by(Appointment.start_time.desc()).all()
        
        return [{
            "id": app.id,
            "start_time": app.start_time,
            "end_time": app.end_time,
            "doctor": {
                "id": app.doctor_id,
                "name": app.doctor_name,
                "type": app.doctor_type
            },
            "status": "upcoming" if app.start_time > current_time else "past"
        } for app in appointments]

@app.get("/my_patients", response_model=list)
def get_my_patients(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> list:
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    doctor_user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not doctor_user or doctor_user.role != "doctor":
        raise HTTPException(status_code=403, detail="User is not a doctor")
    
    doctor = db.query(DoctorsInDB).filter(DoctorsInDB.email == doctor_user.email).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    # Получаем запросы и связанных с ними пациентов
    patient_requests = db.query(UserRequest).filter(
        UserRequest.doctor_id == doctor.id,
        UserRequest.status == True
    ).all()

    # Собираем полную информацию о каждом пациенте
    patients_info = []
    for req in patient_requests:
        patient = db.query(UserInDB).filter(UserInDB.id == req.user_id).first()
        if patient:
            patients_info.append({
                "id": patient.id,
                "name": patient.name,
                "email": patient.email,
                "gender": patient.gender,
                "dateOfBirth": patient.dateOfBirth,
                "phone": patient.phone,
                "address": patient.address,
                "condition": patient.condition,
                "riskLevel": patient.riskLevel,
                "lastVisit": patient.lastVisit,
                "bloodType": patient.bloodType,
                "request": {
                    "id": req.id,
                    "symptoms": req.symptoms,
                    "response": req.response
                }
            })

    return patients_info

@app.get("/doctors/", response_model=list)
def get_all_doctors(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> list:
    payload = verify_access_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    
    doctors = db.query(DoctorsInDB).all()
    
    if not doctors:
        raise HTTPException(status_code=404, detail="No doctors found")
    
    return [{"id": doctor.id, "name": doctor.name, "doctor_type": doctor.doctor_type, "experience": doctor.experience, "rating": doctor.rating, "patient_count": doctor.patient_count} for doctor in doctors]

@app.get("/patient_request/{request_id}", response_model=UserRequestModel)
def get_patient_request(request_id: int, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> UserRequestModel:
    # Проверка токена доктора
    payload = verify_access_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    
    # Находим пользователя по email
    doctor_user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not doctor_user or doctor_user.role != "doctor":
        raise HTTPException(status_code=403, detail="User is not a doctor")
    
    # Получаем доктора из базы данных
    doctor = db.query(DoctorsInDB).filter(DoctorsInDB.email == doctor_user.email).first()
    
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    # Получаем запрос по ID
    patient_request = db.query(UserRequest).filter(UserRequest.id == request_id, UserRequest.doctor_id == doctor.id).first()

    if not patient_request:
        raise HTTPException(status_code=404, detail="Request not found or not associated with this doctor")

    return UserRequestModel.from_orm(patient_request)



@app.put("/patient_request/{request_id}/status/")
def update_patient_request_status(
    request_id: int,
    new_status: bool,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    payload = verify_access_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    
    doctor_user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not doctor_user or doctor_user.role != "doctor":
        raise HTTPException(status_code=403, detail="User is not a doctor")
    
    
    doctor = db.query(DoctorsInDB).filter(DoctorsInDB.email == doctor_user.email).first()
    
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    patient_request = db.query(UserRequest).filter(UserRequest.id == request_id, UserRequest.doctor_id == doctor.id).first()

    if not patient_request:
        raise HTTPException(status_code=404, detail="Request not found or not associated with this doctor")

    
    patient_request.status = False  # Здесь new_status может быть True или False
    db.commit()  # Сохраняем изменения в базе данных

    return {"msg": "Request status updated successfully", "new_status": new_status}




class AppointmentCreate(BaseModel):
    start_time: str
    end_time: str
    doctor_id: int

@app.post("/create_appointment/", response_model=dict)
async def create_appointment(
    appointment: AppointmentCreate, 
    token: str = Depends(oauth2_scheme), 
    db: Session = Depends(get_db)
):
    # Verify token for authorization
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    
    patient = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    doctor = db.query(DoctorsInDB).filter(DoctorsInDB.id == appointment.doctor_id).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    try:
        # Use datetime.strptime instead of fromisoformat
        start_time = datetime.datetime.strptime(appointment.start_time, "%Y-%m-%dT%H:%M:%S")
        end_time = datetime.datetime.strptime(appointment.end_time, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DDTHH:MM:SS")

    new_appointment = Appointment(
        start_time=start_time,
        end_time=end_time,
        patient_name=patient.name,
        patient_id=patient.id,
        doctor_name=doctor.name,
        doctor_id=doctor.id,
        doctor_type=doctor.doctor_type
    )

    db.add(new_appointment)
    db.commit()
    db.refresh(new_appointment)

    return {
        "id": new_appointment.id,
        "message": "Appointment created successfully",
        "start_time": new_appointment.start_time,
        "end_time": new_appointment.end_time,
        "doctor": doctor.name,
        "patient": patient.name
    }


import openai
import shutil
from fastapi import FastAPI, Depends, HTTPException, Form, File, UploadFile, APIRouter
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv
import io

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
@app.post("/predict/")
async def submit_request1(
    symptoms: str = Form(...),
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    # Проверка токена пользователя
    payload = verify_access_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    
    if not user_email:
        raise HTTPException(status_code=400, detail="User email not found in token")
    
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    

    # Сохранение файла на сервере
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Формируем запрос для OpenAI
    prompt = (
        f"Предположи болезнь по данным которые я скинул, результаты не мои, "
        f"ПРОСТО НАПИШИ ДИАГНОЗ И почему так! Основываясь на этих симптомах: {symptoms}"
    )

    # Отправка запроса к OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Замените на нужную модель
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    

    gemini_response = response.choices[0].message['content']

    return {"response": gemini_response}


def file_from_trascript(text):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(
        history=[
        ]
    )

    response = chat_session.send_message(f'Я сделал транскрипт разговора врача и пациента, теперь ты должен из этого трансрипта взять полезную информацию о болезне пациенте, все его симптомы, о его лечении о рекомендациях врача, и так далее, ничего сам не добавляй, сделай прям медицинское описание из поликлиники, бери информацию только из транскрипта: {text}')
    response1 = chat_session.send_message(f'Я сделал транскрипт разговора врача и пациента, теперь ты должен из этого трансрипта взять полезную информацию о болезне пациенте, и напиши рецепт лекарств для пациента исходя от разговора врача: {text}')  
    return (response.text, response1.text)




@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    payload = verify_access_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    
    doctor_user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not doctor_user or doctor_user.role != "doctor":
        raise HTTPException(status_code=403, detail="User is not a doctor")
    
    try:
        audio_bytes = await file.read()

        
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = file.filename

        transcription = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            language="ru"
        )
        
        document_text, reciept_text = file_from_trascript(transcription['text'])

        # Save the transcript and document in the database
        new_document = MedicalDocument(
            doctor_id=doctor_user.id,
            transcript=transcription['text'],
            document=document_text
        )
        db.add(new_document)
        db.commit()

        new_document = MedicalDocument(
            doctor_id=doctor_user.id,
            transcript=transcription['text'],
            document=reciept_text
        )
        db.add(new_document)
        db.commit()
        return {"transcript": transcription['text'], "document": document_text, "reciept": reciept_text}
    
    except Exception as e:
        return {"error": str(e)}


import json
import requests
from fastapi import Depends, HTTPException, UploadFile, File, APIRouter
from sqlalchemy.orm import Session
import datetime


class ChatbotRequest(BaseModel):
    user_message: str
@app.post("/chatbot/")
async def chatbot_interaction(
    request: ChatbotRequest,  # Accept the request body as a Pydantic model
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    # Verify access token
    first_promt = "ты будешь посредником между врачом и пациентом, ты щас разговариваешь с пациентом, и при этом ты должен задать ему дополниетльны евопросы о его болезни, что бы потом отправить этот разговор тебя и пациента врачу, и что бы врач сразу это прочитал и понял в чем проблема, В КОНЦЕ ЕСЛИ У ТЕБЯ НЕ ОСТНЕТСЯ ВОПРОСОВ, НАПИШИ ПАЦИЕНТУ ЧТО ЗАПИШЕШЬ ЕГО НА АНАЛИЗЫ, И НПИШИ АНАЛИЗЫ КОТОРЫЕ ЕМУ НАДО СДАТЬ, смотри я все время буду отправлять тебе история чата, и по этому когда ты видишь что все основные вопросы заданы и ничего больше задавать не надо, то записываешь нас куда надо, и первым словом отправляешь 12345678!!!"
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Fetch previous conversation
    conversation = db.query(ChatbotConversation).filter(ChatbotConversation.user_id == user.id).first()
    mainsymptom = db.query(MainSymptom).filter(MainSymptom.user_id == user.id).first()
    
    # Prepare chat history
    chat_history = []
    mainsymptom_str = ""
    if conversation:
        chat_history = json.loads(conversation.chat_history)
    if mainsymptom:
        mainsymptom_str = mainsymptom.text

    # Add user message to chat history
    chat_history.append({"role": "user", "text": request.user_message})
    mainsymptom_str += " " + request.user_message

    # Prepare prompt by including entire chat history
    prompt = first_promt + "\n".join([f"{entry['role']}: {entry['text']}" for entry in chat_history])

    # Send request to Gemini
    api_key = "AIzaSyBNZ9RJAIcuuLlhCj8KtbxoC6opxY_5q5E"
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, json=data, headers=headers)
    response_data = response.json()
    print(response_data)
    # Extract bot's reply
    bot_reply = response_data['candidates'][0]['content']['parts'][0]['text'].strip()

    # Add bot's response to chat history
    chat_history.append({"role": "bot", "text": bot_reply})

    # Save new message to the Message table
    new_message = Message(
        user_id=user.id,
        user_message=request.user_message,
        bot_reply=bot_reply
    )
    db.add(new_message)

    # Update MainSymptom or create a new entry
    if mainsymptom:
        mainsymptom.text = mainsymptom_str + '\n' + str(bot_reply) + '\n'
    else:
        new_mainsymptom = MainSymptom(text=mainsymptom_str, user_id=user.id)
        db.add(new_mainsymptom)

    # Update or create conversation in the database
    if conversation:
        conversation.chat_history = json.dumps(chat_history)
    else:
        new_conversation = ChatbotConversation(user_id=user.id, chat_history=json.dumps(chat_history))
        db.add(new_conversation)

    db.commit()

    # If "12345678" is found in the bot's reply, trigger submit_request
    if "12345678" in bot_reply:
        await submit_request(symptoms=mainsymptom_str, token=token, db=db)
    
    return {"bot_reply": bot_reply}



@app.get("/messages/")
async def get_messages(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    # Verify the access token and get the user information
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")

    user_email = payload.get("sub")
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Fetch all messages for the authenticated user
    messages = db.query(Message).filter(Message.user_id == user.id).all()

    return messages


def clear_all_tables(db: Session):
    # Удаление всех записей из таблицы MainSymptom
    db.query(MainSymptom).delete()
    
    # Удаление всех записей из таблицы ChatbotConversation
    db.query(ChatbotConversation).delete()
    
    # Удаление всех записей из таблицы Message
    db.query(Message).delete()
    
    # Удаление всех записей из таблицы UserRequest
    db.query(UserRequest).delete()
    
    # Фиксируем изменения
    db.commit()

@app.post("/clear_all/")
async def clear_all_data(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    # Проверка токена
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    # Очищаем все таблицы
    clear_all_tables(db)
    
    return {"message": "Все таблицы успешно очищены"}
