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
import json
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
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    
    best_doctor_json, gemini_response = send_request_to_gemini(
        symptoms=symptoms, 
        api_key="AIzaSyBNZ9RJAIcuuLlhCj8KtbxoC6opxY_5q5E", 
        db=db
    )
    
    best_doctor = best_doctor_json['candidates'][0]['content']['parts'][0]['text'].strip()
    doctor_name, chat_title, color = best_doctor.split()  # Примерный формат выхода от модели

    doctor = db.query(DoctorsInDB).filter(DoctorsInDB.doctor_type == doctor_name).first()

    if not doctor:
        raise HTTPException(status_code=404, detail=f"Doctor {doctor_name} not found")

    # Создаем запрос пользователя
    user_request = UserRequest(
        user_id=user.id,
        name=user.name,
        image_path="uploads",
        symptoms=symptoms,
        response=gemini_response,
        doctor_name=doctor.name,
        color=color,
        chat_title=chat_title,
        status=True,
        doctor_id=doctor.id
    )
    db.add(user_request)
    db.commit()
    db.refresh(user_request)

    # Создаём начальный чат связанный с этим реквестом
    initial_chat = [
        {"role": "bot", "text": f"Здравствуйте! Я ваш медицинский ассистент. Вы описали следующие симптомы: {symptoms}. Расскажите, пожалуйста, подробнее о вашем состоянии и как давно появились эти симптомы?"}
    ]
    
    chatbot_conversation = ChatbotConversation(
        user_id=user.id,
        chat_history=json.dumps(initial_chat),
        request_id=user_request.id
    )
    db.add(chatbot_conversation)
    db.commit()

    return {
        "msg": "Request submitted successfully", 
        "gemini_response": gemini_response,
        "request_id": user_request.id,
        "chat_id": chatbot_conversation.id
    }


@app.get("/my_requests/", response_model=list)
def get_user_requests(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> list:
    payload = verify_access_token(token)
    user_email = payload.get("sub")
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    requests = db.query(UserRequest).filter(UserRequest.user_id == user.id).all()
    return [{"title": req.chat_title.replace("-", " "), "color": req.color, "created_at": req.created_at, "symptoms": req.symptoms, "response": req.response} for req in requests]

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
        
        result = []
        for app in appointments:
            appointment_data = {
                "id": app.id,
                "start_time": app.start_time,
                "end_time": app.end_time,
                "patient": {
                    "id": app.patient_id,
                    "name": app.patient_name
                },
                "appointment_type": app.appointment_type,
                "status": "upcoming" if app.start_time > current_time else "past"
            }
            
            if app.meeting_link:
                appointment_data["meeting_link"] = app.meeting_link
                
            result.append(appointment_data)
        
        return result
    
    # Если пользователь - пациент
    else:
        appointments = db.query(Appointment).filter(
            Appointment.patient_id == user.id
        ).order_by(Appointment.start_time.desc()).all()
        
        result = []
        for app in appointments:
            appointment_data = {
                "id": app.id,
                "start_time": app.start_time,
                "end_time": app.end_time,
                "doctor": {
                    "id": app.doctor_id,
                    "name": app.doctor_name,
                    "type": app.doctor_type
                },
                "appointment_type": app.appointment_type,
                "status": "upcoming" if app.start_time > current_time else "past"
            }
            
            if app.meeting_link:
                appointment_data["meeting_link"] = app.meeting_link
                
            result.append(appointment_data)
        
        return result

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
    start_time: datetime
    end_time: datetime
    doctor_id: int
    appointment_type: str

@app.post("/create_appointment/", response_model=dict)
async def create_appointment(
    appointment: AppointmentCreate, 
    token: str = Depends(oauth2_scheme), 
    db: Session = Depends(get_db)
):

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

    existing_appointment = (
        db.query(Appointment)
        .filter(
            Appointment.doctor_id == appointment.doctor_id,
            Appointment.start_time < appointment.end_time,
            Appointment.end_time > appointment.start_time
        )
        .first()
    )
    if existing_appointment:
        raise HTTPException(status_code=400, detail="This time slot is already booked for the doctor")

    meeting_link = None
    if appointment.appointment_type.lower() == "online":
        meeting_link = "https://meet.google.com/zpx-ftcy-oxi"
    
    new_appointment = Appointment(
        start_time=appointment.start_time,
        end_time=appointment.end_time,
        patient_name=patient.name,
        patient_id=patient.id,
        doctor_name=doctor.name,
        doctor_id=doctor.id,
        doctor_type=getattr(doctor, "doctor_type", "Unknown"),
        appointment_type=appointment.appointment_type,
        meeting_link=meeting_link
    )

    db.add(new_appointment)
    db.commit()
    db.refresh(new_appointment)

    response = {
        "id": new_appointment.id,
        "message": "Appointment created successfully",
        "start_time": new_appointment.start_time,
        "end_time": new_appointment.end_time,
        "doctor": doctor.name,
        "patient": patient.name,
        "appointment_type": new_appointment.appointment_type
    }
    
    if meeting_link:
        response["meeting_link"] = meeting_link
        
    return response


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
@app.post("/chatbot/{chat_id}")
async def chatbot_interaction(
    chat_id: int,
    request: ChatbotRequest,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    # Verify access token
    first_promt = "ты будешь посредником между врачом и пациентом, ты щас разговариваешь с пациентом, и при этом ты должен задать ему дополниетльны евопросы о его болезни, что бы потом отправить этот разговор тебя и пациента врачу, и что бы врач сразу это прочитал и понял в чем проблема. Никогда не упоминай что ты AI или что передашь информацию врачу. Просто веди диалог как медицинский ассистент. В конце спроси еще вопросы если их останутся."
    
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Получаем конкретный чат по ID
    conversation = db.query(ChatbotConversation).filter(
        ChatbotConversation.id == chat_id,
        ChatbotConversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Получаем реквест, связанный с этим чатом
    user_request = db.query(UserRequest).filter(
        UserRequest.id == conversation.request_id
    ).first()
    
    if not user_request:
        raise HTTPException(status_code=404, detail="Associated request not found")
    
    # Prepare chat history
    chat_history = json.loads(conversation.chat_history)
    
    # Add user message to chat history
    chat_history.append({"role": "user", "text": request.user_message})
    
    # Prepare prompt by including entire chat history
    prompt = first_promt + "\n\nСимптомы пациента: " + user_request.symptoms + "\n\n" + "\n".join([f"{entry['role']}: {entry['text']}" for entry in chat_history])
    
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
    
    # Extract bot's reply
    bot_reply = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
    
    # Add bot's response to chat history
    chat_history.append({"role": "bot", "text": bot_reply})
    
    # Save message to database
    new_message = Message(
        user_id=user.id,
        user_message=request.user_message,
        bot_reply=bot_reply
    )
    db.add(new_message)
    
    # Update conversation in the database
    conversation.chat_history = json.dumps(chat_history)
    db.commit()
    
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

@app.get("/my_chats/", response_model=List[dict])
async def get_user_chats(
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
    
    # Получаем все чаты с связанными реквестами
    chats = db.query(ChatbotConversation, UserRequest).join(
        UserRequest, ChatbotConversation.request_id == UserRequest.id
    ).filter(
        ChatbotConversation.user_id == user.id
    ).all()
    
    return [{
        "chat_id": chat.id,
        "request_id": request.id,
        "title": request.chat_title,
        "color": request.color,
        "doctor_name": request.doctor_name,
        "created_at": chat.created_at,
        "last_message": json.loads(chat.chat_history)[-1]["text"][:50] + "..." if chat.chat_history else ""
    } for chat, request in chats]

@app.get("/chat/{chat_id}", response_model=dict)
async def get_chat_history(
    chat_id: int,
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
    
    # Получаем конкретный чат по ID
    conversation = db.query(ChatbotConversation).filter(
        ChatbotConversation.id == chat_id,
        ChatbotConversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Получаем реквест, связанный с этим чатом
    user_request = db.query(UserRequest).filter(
        UserRequest.id == conversation.request_id
    ).first()
    
    # Возвращаем историю чата и информацию о реквесте
    return {
        "chat_id": conversation.id,
        "request_id": user_request.id,
        "doctor_name": user_request.doctor_name,
        "title": user_request.chat_title,
        "color": user_request.color,
        "created_at": conversation.created_at,
        "chat_history": json.loads(conversation.chat_history)
    }
