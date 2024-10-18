from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from models import *
from config import get_db, init_db
from auth_utils import hash_password, verify_password, create_access_token, verify_access_token
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from datetime import timedelta
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
import requests
from config import Base
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware

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
@app.post("/register/", response_model=dict)
def register_user(user: UserCreate, db: Session = Depends(get_db)) -> dict:
    db_user = db.query(UserInDB).filter(UserInDB.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    if user.role not in ["doctor", "patient"]:
        raise HTTPException(status_code=400, detail="Invalid role, must be 'doctor' or 'patient'")

    hashed_pw = hash_password(user.password)
    new_user = UserInDB(
        name=user.name,
        email=user.email,
        hashed_password=hashed_pw,
        role=user.role
    )
    db.add(new_user)
    
    # Сохраняем доктора в отдельной таблице, если это доктор
    if user.role == "doctor":
        new_doctor = DoctorsInDB(
            name=user.name,
            email=user.email,
            hashed_password=hashed_pw,
            doctor_type=user.doctor_type  # Сохраняем тип доктора
        )
        db.add(new_doctor)
    
    db.commit()
    return {"msg": "User created successfully"}



@app.post("/login/", response_model=Token)
def login_for_access_token(user: UserLogin, db: Session = Depends(get_db)) -> Token:
    db_user = db.query(UserInDB).filter(UserInDB.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.email, "role": db_user.role},  # Добавляем роль в токен
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "role": db_user.role}


UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def send_request_to_gemini(symptoms: str, api_key: str, db: Session) -> str:
    # Получение списка докторов из базы данных

    doctors = db.query(DoctorsInDB).all()
    
    # Создаем строку с именами докторов
    doctor_names = ', '.join([doctor.doctor_type for doctor in doctors])
    
    # Запрос для Gemini API для анализа симптомов
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
    data = {
        "contents": [
            {
                "parts": [
                    {"text": symptoms}
                ]
            }
        ]
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, json=data, headers=headers)
    print(1)
    # Формируем новый промпт для выбора врача
    new_prompt = f"ЩАС ПРОСТО ОТПРАВЬ МНЕ ОДНО СЛОВО, НИЧЕГО БОЛЬШЕ НЕ ПИШИ, ПРОСТО ВЫБЕРИ ВРАЧА ИЗ СПИСКА К КОТОРОМУ ИДТИ: {doctor_names}, ДАЖЕ ТОЧКУ НЕ ПИШИ. Симптомы: {symptoms}"
    data = {
        "contents": [
            {
                "parts": [
                    {"text": new_prompt}
                ]
            }
        ]
    }
    best_doctor = requests.post(url, json=data, headers=headers)
    
    # print(best_doctor.text)
    best_doctor_json = best_doctor.json()
    gemini_response_json = response.json()

    if response.status_code == 200:
        return best_doctor_json, gemini_response_json
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Error from Gemini API: {response.text}")

@app.post("/submit_request/")
async def submit_request(
    symptoms: str = Form(...),
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    # Проверка токена пользователя
    payload = verify_access_token(token)
    
    # Проверяем, что payload не None
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token or unauthorized")
    
    user_email = payload.get("sub")
    
    if not user_email:
        raise HTTPException(status_code=400, detail="User email not found in token")
    
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    
    # Сохранение изображения на сервере
    file_location = f"{UPLOAD_FOLDER}{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Отправка запроса на API Gemini
    best_doctor = ""
    gemini_response = ""
    try:
        best_doctor1, gemini_response1 = send_request_to_gemini(symptoms=symptoms, api_key="AIzaSyBNZ9RJAIcuuLlhCj8KtbxoC6opxY_5q5E", db=db)
        print(type(best_doctor1))
        print(type(gemini_response1))
        best_doctor = best_doctor1['candidates'][0]['content']['parts'][0]['text'].strip()
        gemini_response = gemini_response1['candidates'][0]['content']['parts'][0]['text'].strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    print(best_doctor1)
    print(gemini_response1)
    # Найти доктора в базе данных по его имени
    doctor = db.query(DoctorsInDB).filter(DoctorsInDB.doctor_type == best_doctor).first()


    # Проверка, что доктор найден
    if not doctor:
        raise HTTPException(status_code=404, detail=f"Doctor {best_doctor} not found")

    # Присваиваем doctor_id
    best_doctor_id = doctor.id
    best_doctor_name = doctor.name
    # Сохранение запроса в базе данных
    user_request = UserRequest(
        user_id=user.id,
        name = user.name,
        image_path=file_location,
        symptoms=symptoms,
        response=gemini_response,
        doctor_name = best_doctor_name,
        doctor_id=best_doctor_id
    )
    db.add(user_request)
    db.commit()

    return {"msg": "Request submitted successfully", "gemini_response": gemini_response}


@app.get("/my_requests/", response_model=list)
def get_user_requests(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> list:
    # Проверка токена пользователя
    payload = verify_access_token(token)
    user_email = payload.get("sub")
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    # Получение запросов пользователя
    requests = db.query(UserRequest).filter(UserRequest.user_id == user.id).all()
    return [{"symptoms": req.symptoms, "image_path": req.image_path, "response": req.response} for req in requests]



@app.get("/my_patients/", response_model=list)
def get_my_patients(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> list:
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

    # Получаем все запросы, связанные с доктором
    patient_requests = db.query(UserRequest).filter(UserRequest.doctor_id == doctor.id).all()

    return [{"name": req.name, "id": req.id, "image_path": req.image_path, "symptoms": req.symptoms, "response": req.response} for req in patient_requests]

