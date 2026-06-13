from pydantic import BaseModel


class Student(BaseModel):
    name: str
    age: int


new_Student = {'name':"abhi", "age": 22}

student = Student(**new_Student)

print(student)