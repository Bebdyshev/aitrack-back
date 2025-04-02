from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from models import *
from config import get_db, init_db
from auth_utils import hash_password, verify_password, create_access_token, verify_access_token
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, List, Any
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

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_model_name = os.getenv("GEMINI_MODEL_NAME")
genai.configure(api_key=gemini_api_key)

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
  model_name=gemini_model_name,
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
        model_name=gemini_model_name,
        generation_config=generation_config
    )

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(symptoms)

    # Формируем новый запрос для выбора врача
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
    
    new_prompt = (
        f"CHOOSE A DOCTOR FROM THE LIST BASED ON THE GIVEN SYMPTOMS: {doctor_names}. "
        f"PICK A DOCTOR AND CREATE A VERY SHORT CHAT TITLE (e.g., 'Facial-Irritation') and color for chat pfp. Symptoms: {symptoms} "
        f"OUTPUT EXACTLY IN THIS FORMAT WITHOUT ANY ADDITIONAL TEXT: [doctor-type] [chat-title] rgb(r,g,b)"
        f"Example output: Dermatologist Facial-Irritation rgb(255,100,100)"
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
        print(best_doctor_json)
        print(response.text)
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
        api_key=gemini_api_key, 
        db=db
    )
    
    best_doctor = best_doctor_json['candidates'][0]['content']['parts'][0]['text'].strip()
    print("Gemini response:", best_doctor)  # Для отладки
    
    try:
        # Используем более надежный способ разбора ответа
        parts = best_doctor.split('rgb')
        doctor_info = parts[0].strip()
        color = 'rgb' + parts[1].strip()
        print(doctor_info)
        # Разделяем информацию о докторе на тип и название чата
        doctor_parts = doctor_info.split()
        doctor_name = doctor_parts[0]
        chat_title = doctor_parts[1]  # Все слова между типом доктора и цветом
        
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
            {"role": "bot", "text": f"Hello! I am your medical assistant. You described the following symptoms: {symptoms}. Please tell me more about your condition and how long ago these symptoms appeared?"}
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

    except Exception as e:
        print(f"Error parsing Gemini response: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Error processing doctor selection. Please try again."
        )


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
print(os.getenv("OPENAI_API_KEY"))
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
    
    # Получаем переменные окружения для Azure OpenAI
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_KEY")

    # Сохранение файла на сервере
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prompt = (
        f"Based on the information I've shared (which is not about me personally), "
        f"please provide a potential diagnosis and explain your reasoning. "
        f"Consider these symptoms: {symptoms}. "
        f"ONLY PROVIDE THE DIAGNOSIS AND EXPLANATION without any additional information."
    )

    # Формируем запрос для Azure OpenAI API
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_api_key
    }
    
    data = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    
    try:
        # Отправляем запрос к Azure OpenAI API
        response = requests.post(
            azure_endpoint,
            headers=headers,
            json=data
        )
        
        # Проверяем статус ответа
        response.raise_for_status()
        
        # Получаем ответ
        response_data = response.json()
        
        if 'choices' in response_data and len(response_data['choices']) > 0:
            gemini_response = response_data['choices'][0]['message']['content']
            
            # Создаем структурированный ответ
            return {
                "status": "success",
                "data": {
                    "file": {
                        "name": file.filename,
                        "location": file_location,
                        "type": file.content_type
                    },
                    "symptoms": symptoms,
                    "analysis": {
                        "raw_response": gemini_response,
                        "timestamp": datetime.datetime.now().isoformat()
                    },
                    "user": {
                        "id": user.id,
                        "email": user.email
                    }
                }
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail={
                    "status": "error",
                    "message": "No valid response from Azure OpenAI API",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        
    except requests.exceptions.RequestException as e:
        error_response = {
            "status": "error",
            "error": {
                "type": "API_ERROR",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
        print(f"Error calling Azure OpenAI API: {str(e)}")
        raise HTTPException(status_code=500, detail=error_response)
        
    except Exception as e:
        error_response = {
            "status": "error",
            "error": {
                "type": "INTERNAL_ERROR",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=error_response)


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

    chat_session = model.start_chat(history=[])

    medical_doc_prompt = f"""As an AI medical assistant, analyze this doctor-patient conversation transcript 
    and create a structured medical document that will help the doctor make accurate diagnosis and treatment decisions.
    
    Focus on extracting and organizing:
    - Patient's reported symptoms and their duration
    - Relevant medical history mentioned
    - Physical examination findings
    - Doctor's observations and preliminary diagnosis
    - Any test results or measurements discussed
    
    Format this as a professional medical record, similar to clinical documentation. 
    Only include information explicitly mentioned in the transcript: {text}"""

    prescription_prompt = f"""Based on the same doctor-patient conversation, create a clear summary of:
    - Prescribed medications with their dosages and duration
    - Treatment recommendations
    - Lifestyle modifications suggested
    - Follow-up instructions
    
    This will serve as a reference for both the doctor and patient to ensure proper treatment adherence.
    Only include medications and recommendations explicitly mentioned by the doctor in the transcript: {text}"""

    response = chat_session.send_message(medical_doc_prompt)
    response1 = chat_session.send_message(prescription_prompt)
    
    return (response.text, response1.text)



import openai

@app.post("/transcribe")
async def transcribe_audio(
    patient_id: int,
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

    patient = db.query(UserInDB).filter(
        UserInDB.id == patient_id,
        UserInDB.role == "patient"
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)
        
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_KEY")
        
        url = f"{azure_endpoint}/openai/deployments/whisper/audio/translations?api-version=2024-06-01"
        
        files = {
            'file': (file.filename, audio_file, file.content_type)
        }
        
        headers = {
            "api-key": azure_api_key
        }
        
        response = requests.post(url, files=files, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, 
                                detail=f"Azure OpenAI API error: {response.text}")
        
        result = response.json()
        transcription_text = result.get("text", "")
        
        document_text, receipt_text = file_from_trascript(transcription_text)

        new_document = MedicalDocument(
            doctor_id=doctor_user.id,
            patient_id=patient_id,
            transcript=transcription_text,
            document=document_text,
            receipt=receipt_text
        )
        db.add(new_document)
        db.commit()

        patient.lastVisit = datetime.datetime.now()
        db.commit()

        return {
            "transcript": transcription_text,
            "document": document_text,
            "receipt": receipt_text,
            "document_id": new_document.id
        }

    except Exception as e:
        db.rollback()
        print(f"Error in transcribe_audio: {str(e)}")
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

    first_prompt = """You are a medical assistant chatting with a patient. Ask them additional questions about their symptoms 
    to better understand their condition. Don't mention that you're AI or that you'll send this conversation to a doctor. 
    Just have a dialogue as a medical assistant would. Response in english.
    IF YOU DONT HAVE ANY QUESTIONS, END YOUR RESPONSE WITH THE EXACT PHRASE: 'END_OF_CONSULTATION_MARKER'
    """
    
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get the specific chat by ID
    conversation = db.query(ChatbotConversation).filter(
        ChatbotConversation.id == chat_id,
        ChatbotConversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Get the request associated with this chat
    user_request = db.query(UserRequest).filter(
        UserRequest.id == conversation.request_id
    ).first()
    
    if not user_request:
        raise HTTPException(status_code=404, detail="Associated request not found")
    
    # Получаем информацию о докторе из запроса
    doctor = db.query(DoctorsInDB).filter(DoctorsInDB.id == user_request.doctor_id).first()
    doctor_type = doctor.doctor_type if doctor else "Unknown"
    doctor_id = doctor.id
    # Prepare chat history
    chat_history = json.loads(conversation.chat_history)
    
    # Add user message to chat history
    chat_history.append({"role": "user", "text": request.user_message})
    
    # Prepare prompt by including entire chat history
    prompt = first_prompt + "\n\nPatient symptoms: " + user_request.symptoms + "\n\n" + "\n".join([f"{entry['role']}: {entry['text']}" for entry in chat_history])
    
    # Send request to Gemini
    api_key = gemini_api_key
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
    
    original_bot_reply = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
    
    consultation_complete = "END_OF_CONSULTATION_MARKER" in original_bot_reply
    
    bot_reply = original_bot_reply.strip()
    
    chat_history.append({"role": "bot", "text": bot_reply})
    
    new_message = Message(
        user_id=user.id,
        user_message=request.user_message,
        bot_reply=bot_reply
    )
    db.add(new_message)
    
    conversation.chat_history = json.dumps(chat_history)
    db.commit()
    
    return {
        "bot_reply": bot_reply,
        "consultation_complete": consultation_complete,
        "doctor_type": doctor_type,
        "doctor_name": user_request.doctor_name,
        "request_id": user_request.id,
        "doctor_id": doctor_id
    }

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
        "doctor_id": request.doctor_id,
        "created_at": chat.created_at,
        "confirmed": chat.confirmed,
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
    
    conversation = db.query(ChatbotConversation).filter(
        ChatbotConversation.id == chat_id,
        ChatbotConversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    user_request = db.query(UserRequest).filter(
        UserRequest.id == conversation.request_id
    ).first()
    
    return {
        "chat_id": conversation.id,
        "request_id": user_request.id,
        "doctor_name": user_request.doctor_name,
        "doctor_id": user_request.doctor_id,
        "title": user_request.chat_title,
        "color": user_request.color,
        "created_at": conversation.created_at,
        "confirmed": conversation.confirmed,
        "chat_history": json.loads(conversation.chat_history)
    }

@app.post("/chat_confirm/{chat_id}", response_model=dict)
async def confirm_chat_and_create_appointment(
    chat_id: int,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Mark chat as confirmed and ready for appointment scheduling.
    This endpoint is called when a consultation is complete and the patient is ready to schedule an appointment.
    """
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get the specific chat by ID
    conversation = db.query(ChatbotConversation).filter(
        ChatbotConversation.id == chat_id,
        ChatbotConversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Get the request associated with this chat
    user_request = db.query(UserRequest).filter(
        UserRequest.id == conversation.request_id
    ).first()
    
    if not user_request:
        raise HTTPException(status_code=404, detail="Associated request not found")
    
    # Получаем информацию о докторе из запроса
    doctor = db.query(DoctorsInDB).filter(DoctorsInDB.id == user_request.doctor_id).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    # Mark conversation as confirmed (add confirmed field if it doesn't exist)
    if not hasattr(conversation, 'confirmed'):
        # You need to add this field to your ChatbotConversation model
        # ALTER TABLE chatbot_conversations ADD COLUMN confirmed BOOLEAN DEFAULT FALSE;
        pass
    
    conversation.confirmed = True
    
    # You might also want to add a confirmation message to the chat history
    chat_history = json.loads(conversation.chat_history)
    confirmation_message = {
        "role": "system", 
        "text": "Consultation completed. Your information has been recorded and is ready for appointment scheduling."
    }
    chat_history.append(confirmation_message)
    conversation.chat_history = json.dumps(chat_history)
    
    # Save changes
    db.commit()
    
    return {
        "success": True,
        "message": "Chat confirmed and ready for appointment",
        "chat_id": chat_id,
        "request_id": user_request.id,
        "doctor_id": doctor.id,
        "doctor_name": doctor.name,
        "doctor_type": doctor.doctor_type
    }

@app.get("/patient_info/{patient_id}", response_model=Dict[str, Any])
async def get_patient_info(
    patient_id: int,
    include_chat_history: bool = True,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive information about a specific patient.
    Only accessible to doctors.
    
    Parameters:
    - patient_id: ID of the patient
    - include_chat_history: Whether to include full chat history (default: True)
    """
    # Verify token and ensure the user is a doctor
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    doctor_user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not doctor_user or doctor_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access patient information")
    
    # Verify the doctor exists in the doctors table
    doctor = db.query(DoctorsInDB).filter(DoctorsInDB.email == doctor_user.email).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor profile not found")
    
    # Get the patient information
    patient = db.query(UserInDB).filter(UserInDB.id == patient_id, UserInDB.role == "patient").first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get patient requests associated with this doctor
    patient_requests = db.query(UserRequest).filter(
        UserRequest.user_id == patient.id,
        UserRequest.doctor_id == doctor.id
    ).all()
    
    # Get patient appointments with this doctor
    appointments = db.query(Appointment).filter(
        Appointment.patient_id == patient.id,
        Appointment.doctor_id == doctor.id
    ).order_by(Appointment.start_time.desc()).all()
    
    # Get associated chatbot conversations
    conversations = []
    for req in patient_requests:
        conv = db.query(ChatbotConversation).filter(
            ChatbotConversation.request_id == req.id
        ).first()
        if conv:
            chat_data = {
                "id": conv.id,
                "request_id": req.id,
                "created_at": conv.created_at,
                "confirmed": conv.confirmed,
                "chat_title": req.chat_title,
                "symptoms": req.symptoms
            }
            
            # Include chat history if requested
            if include_chat_history:
                try:
                    chat_data["chat_history"] = json.loads(conv.chat_history)
                except:
                    chat_data["chat_history"] = []
            
            conversations.append(chat_data)
    
    # Get current time for appointment status
    current_time = datetime.datetime.now()
    
    # Compile the patient information
    patient_info = {
        "id": patient.id,
        "name": patient.name,
        "email": patient.email,
        "role": patient.role,
        "personal_info": {
            "gender": patient.gender,
            "dateOfBirth": patient.dateOfBirth,
            "age": calculate_age(patient.dateOfBirth) if patient.dateOfBirth else None,
            "phone": patient.phone,
            "address": patient.address,
        },
        "medical_info": {
            "condition": patient.condition,
            "riskLevel": patient.riskLevel,
            "lastVisit": patient.lastVisit,
            "bloodType": patient.bloodType,
        },
        "requests": [{
            "id": req.id,
            "date": req.created_at,
            "symptoms": req.symptoms,
            "response": req.response,
            "status": req.status,
            "chat_title": req.chat_title
        } for req in patient_requests],
        "appointments": [{
            "id": app.id,
            "start_time": app.start_time,
            "end_time": app.end_time,
            "appointment_type": app.appointment_type,
            "status": "upcoming" if app.start_time > current_time else "past",
            "meeting_link": app.meeting_link
        } for app in appointments],
        "conversations": conversations
    }
    
    # Try to get the most recent medical documents if available
    try:
        last_medical_docs = db.query(MedicalDocument).filter(
            MedicalDocument.doctor_id == doctor.id,
            MedicalDocument.patient_id == patient.id
        ).order_by(MedicalDocument.created_at.desc()).limit(2).all()
        
        if last_medical_docs:
            patient_info["medical_documents"] = [{
                "id": doc.id,
                "date": doc.created_at,
                "document": doc.document,
                "transcript": doc.transcript
            } for doc in last_medical_docs]
    except Exception as e:
        # If there's an error or medical documents aren't available, just continue
        print(f"Error fetching medical documents: {str(e)}")
    
    return patient_info

# Helper function to calculate age from date of birth
def calculate_age(dob_str):
    if not dob_str:
        return None
    
    try:
        # Parse date string - adjust the format according to your date format
        dob_formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"]
        
        for fmt in dob_formats:
            try:
                dob = datetime.strptime(dob_str, fmt)
                today = datetime.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                return age
            except ValueError:
                continue
        
        return None  # None of the formats worked
    except:
        return None  # Any other error

@app.get("/my_transcribes", response_model=List[dict])
async def get_my_transcribes(
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
    
    try:
        if user.role == "doctor":
            user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
            if not user:
                raise HTTPException(status_code=404, detail="Doctor profile not found")
            
            # Добавляем отладочную информацию
            print(f"Looking for documents for doctor ID: {user.id}")
            
            # Сначала проверим, есть ли вообще документы для этого доктора
            doc_count = db.query(MedicalDocument).filter(
                MedicalDocument.doctor_id == user.id
            ).count()
            print(f"Found {doc_count} documents for this doctor")
            
            # Исправленный запрос
            documents = db.query(MedicalDocument, UserInDB).join(
                UserInDB, 
                UserInDB.id == MedicalDocument.patient_id  # Уточняем условие JOIN
            ).filter(
                MedicalDocument.doctor_id == user.id
            ).order_by(MedicalDocument.created_at.desc()).all()
            
            print(f"After join, found {len(documents)} documents")
            
            result = [{
                "id": doc.id,
                "created_at": doc.created_at,
                "patient": {
                    "id": patient.id,
                    "name": patient.name,
                    "email": patient.email
                },
                "transcript": doc.transcript,
                "document": doc.document,
                "receipt": doc.receipt
            } for doc, patient in documents]
            
            print(f"Returning {len(result)} documents")
            return result
            
    except Exception as e:
        print(f"Error in get_my_transcribes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def clear_specific_records(db: Session, record_id: int):
    try:
        # Удаляем записи из ChatbotConversation
        chat = db.query(ChatbotConversation).filter(
            ChatbotConversation.request_id == record_id
        ).delete()
        
        # Удаляем запрос пользователя
        user_request = db.query(UserRequest).filter(
            UserRequest.id == record_id
        ).delete()
        
        # Фиксируем изменения
        db.commit()
        
        return {
            "status": "success",
            "deleted": {
                "chat_records": chat,
                "user_request": user_request
            }
        }
        
    except Exception as e:
        db.rollback()
        print(f"Error during deletion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during deletion: {str(e)}"
        )

# Эндпоинт для вызова функции
@app.delete("/clear_records/{record_id}")
async def delete_records(
    record_id: int,
    db: Session = Depends(get_db)
):
    
    return clear_specific_records(db, record_id)
