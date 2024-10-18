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

app = FastAPI()

init_db()

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

# Функция для отправки запроса на API Gemini
def send_request_to_gemini(symptoms: str, api_key: str) -> str:
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
    
    if response.status_code == 200:
        return response.text
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
    user_email = payload.get("sub")
    user = db.query(UserInDB).filter(UserInDB.email == user_email).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    # Сохранение изображения на сервере
    file_location = f"{UPLOAD_FOLDER}{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Отправка запроса на API Gemini
    try:
        gemini_response = send_request_to_gemini(symptoms=symptoms, api_key="AIzaSyBNZ9RJAIcuuLlhCj8KtbxoC6opxY_5q5E")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Сохранение запроса в базе данных
    user_request = UserRequest(
        user_id=user.id,
        image_path=file_location,
        symptoms=symptoms,
        response=gemini_response
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


