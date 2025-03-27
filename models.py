from sqlalchemy import Column, Integer, Float, String, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional

Base = declarative_base()


class DoctorsInDB(Base):
    __tablename__ = "doctors"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    doctor_type = Column(String)
    experience = Column(Integer)
    rating = Column(Float)
    patient_count = Column(Integer) 
    

class UserInDB(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String)

    gender = Column(String, nullable=True)
    dateOfBirth = Column(DateTime, nullable=True)
    phone = Column(String, nullable=True)
    address = Column(String, nullable=True)
    condition = Column(String, nullable=True)
    riskLevel = Column(String, nullable=True)
    lastVisit = Column(DateTime, nullable=True)
    bloodType = Column(String, nullable=True)

    requests = relationship("UserRequest", back_populates="user")
    conversations = relationship("ChatbotConversation", back_populates="user")
    messages = relationship("MainSymptom", back_populates="user")
    main_symptoms = relationship("MainSymptom", back_populates="user")


class UserRequest(Base):
    __tablename__ = "user_requests"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_path = Column(String, nullable=False)
    chat_title = Column(String, nullable=False)
    color = Column(String, nullable=False)
    symptoms = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    doctor_id = Column(Text, nullable=True)
    doctor_name = Column(Text, nullable=True)
    status = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("UserInDB", back_populates="requests")
    conversations = relationship("ChatbotConversation", back_populates="request")
    

UserInDB.requests = relationship("UserRequest", back_populates="user")


class Appointment(Base):
    __tablename__ = "appointments"
    
    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    patient_name = Column(String, nullable=False)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    doctor_name = Column(String, nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    doctor_type = Column(String, nullable=False)
    appointment_type = Column(String, nullable=True)  # "online" или "offline"
    meeting_link = Column(String, nullable=True)      # ссылка на видеоконференцию
    
    patient = relationship("UserInDB", back_populates="appointments")
    doctor = relationship("DoctorsInDB", back_populates="appointments")

UserInDB.appointments = relationship("Appointment", back_populates="patient")
DoctorsInDB.appointments = relationship("Appointment", back_populates="doctor")


class MedicalDocument(Base):
    __tablename__ = "medical_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey('doctors.id'))
    patient_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    transcript = Column(Text, nullable=False)
    document = Column(Text, nullable=False)
    receipt = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    doctor = relationship("DoctorsInDB", foreign_keys=[doctor_id])
    patient = relationship("UserInDB", foreign_keys=[patient_id])

class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: str
    # Поля для доктора
    doctor_type: Optional[str] = None
    experience: Optional[int] = None
    rating: Optional[float] = None
    patient_count: Optional[int] = None
    # Поля для пациента
    gender: Optional[str] = None
    dateOfBirth: Optional[datetime] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    condition: Optional[str] = None
    riskLevel: Optional[str] = None
    bloodType: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str


class UserRequestModel(BaseModel):
    id: int
    name: str
    user_id: int
    image_path: str
    symptoms: str
    response: str
    doctor_id: int
    status: bool
    doctor_name: str

    class Config:
        orm_mode = True
        from_attributes = True



class ChatbotConversation(Base):
    __tablename__ = "chatbot_conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    chat_history = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    request_id = Column(Integer, ForeignKey("user_requests.id"), nullable=True)
    confirmed = Column(Boolean, default=False)

    user = relationship("UserInDB", back_populates="conversations")
    request = relationship("UserRequest", back_populates="conversations")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user_message = Column(Text, nullable=False)
    bot_reply = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserInDB", back_populates="messages")


# Update UserInDB to include the relationship
UserInDB.messages = relationship("Message", back_populates="user")


class MainSymptom(Base):
    __tablename__ = "main_symptom"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    user = relationship("UserInDB", back_populates="main_symptoms")
