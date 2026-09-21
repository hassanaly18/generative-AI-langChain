from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Hassan"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=4)

new_student = {"age": '32', "email": 'abc@gmail.com', "cgpa": 2.99}

student = Student(**new_student)

student_json = student.model_dump_json()
print(student)