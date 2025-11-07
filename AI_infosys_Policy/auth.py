# --- auth.py ---

from fastapi import APIRouter, Form, Depends, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError  # Import this to catch errors
from db import SessionLocal
from models import User
import bcrypt
from config import templates  # Import from config.py

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Login Routes ---

@router.get("/login")
async def login_page(request: Request):
    # This route is defined in main.py as "/", but good to have
    # in case you want a dedicated /login URL
    return templates.TemplateResponse("auth/login.html", {"request": request})

@router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()
    
    # Check if user exists AND password is correct
    if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        request.session['user'] = username
        return RedirectResponse(url="/home", status_code=303)
    
    # **FIXED:** On failure, send an error message back to the login page
    return templates.TemplateResponse(
        "auth/login.html", 
        {"request": request, "error": "Invalid username or password"}
    )

# --- Register Route ---

@router.post("/register")
async def register(
    request: Request,  # Add request for returning template on error
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # **FIXED:** Check if user already exists BEFORE trying to create
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        return templates.TemplateResponse(
            "auth/register.html", 
            {"request": request, "error": "Username already taken"}
        )

    # If user does not exist, proceed
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = User(username=username, password=hashed_pw)
    
    db.add(new_user)
    db.commit()
    
    # On success, send to login page
    return RedirectResponse(url="/", status_code=303)

# --- Logout Route (Only one!) ---

@router.get("/logout")
async def logout(request: Request):
    """
    Clears the user session and redirects to the login page.
    """
    request.session.clear()  # This "forgets" the user
    
    # This sends them back to your login page
    return RedirectResponse(url="/", status_code=303)