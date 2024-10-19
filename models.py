from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from sqlalchemy.orm import relationship

Base = declarative_base()


class DoctorsInDB(Base):
    __tablename__ = "doctors"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    doctor_type = Column(String)  


class UserInDB(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String)


class UserRequest(Base):
    __tablename__ = "user_requests"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_path = Column(String, nullable=False)
    symptoms = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    doctor_id = Column(Text, nullable=True)
    doctor_name = Column(Text, nullable=True)
    user = relationship("UserInDB", back_populates="requests")
    

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

    patient = relationship("UserInDB", back_populates="appointments")
    doctor = relationship("DoctorsInDB", back_populates="appointments")

# Не забудьте обновить ваши модели UserInDB и DoctorsInDB, чтобы они знали о связи.
UserInDB.appointments = relationship("Appointment", back_populates="patient")
DoctorsInDB.appointments = relationship("Appointment", back_populates="doctor")



class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: str
    doctor_type: str = None  # Поле для типа доктора (по умолчанию None)

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
    doctor_name: str

    class Config:
        orm_mode = True
        from_attributes = True