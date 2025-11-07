from fastapi.templating import Jinja2Templates

# This is the new, central location for the templates object
templates = Jinja2Templates(directory="templates")