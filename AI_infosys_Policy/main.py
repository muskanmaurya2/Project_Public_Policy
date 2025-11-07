# --- main.py ---
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import RedirectResponse
from config import templates  # Import from config.py

# Assuming you have these router files
from auth import router as auth_router 
from model import router as model_router

app = FastAPI()

# Session Middleware
app.add_middleware(SessionMiddleware, secret_key="your_strong_secret_key_12345", session_cookie="session_id")

# Mount Static Files (only need to do this once)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include Routers from other files
app.include_router(auth_router)
app.include_router(model_router)

# --- Core App Routes ---

@app.get("/")
async def root(request: Request):
    # This is your login page
    return templates.TemplateResponse("auth/login.html", {"request": request})

@app.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse("auth/register.html", {"request": request})


@app.get("/home")
async def home(request: Request):
    
    # The Bouncer checks your ID (your session)
    if "user" not in request.session:
        
        # Bouncer: "You're not on the list. Go to the front door."
        return RedirectResponse(url="/", status_code=303) # <--- REDIRECTS TO LOGIN
    
    # Bouncer: "Welcome, come on in."
    return templates.TemplateResponse("index.html", {"request": request})