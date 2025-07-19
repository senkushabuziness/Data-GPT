#app.py
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from google.cloud import storage
import os

from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests

app = FastAPI()

# ---------- Global user state for testing ----------
user_email = None

# ---------- Google OAuth Setup ----------
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
CLIENT_ID = os.getenv("OAUTH_GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")

REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8080/oauth-callback")

flow = Flow.from_client_config(
    {
        "web": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI]
        }
    },
    scopes=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"]
)
flow.redirect_uri = REDIRECT_URI

# ---------- Routes ----------

@app.get("/")
async def index():
    global user_email
    if user_email:
        # If user is logged in, show upload form
        return HTMLResponse(f"""
            <h2>✅ Logged in as {user_email}</h2>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input type="file" name="file" required>
                <button type="submit">Upload File to GCS</button>
            </form>
        """)
    else:
        # If not logged in, show login button
        return HTMLResponse("""
            <h2>🔐 Please login to continue</h2>
            <a href="/login">Login with Google</a>
        """)

@app.get("/login")
async def login():
    authorization_url, state = flow.authorization_url()
    return RedirectResponse(authorization_url)

@app.get("/oauth-callback")
async def oauth_callback(request: Request):
    global user_email
    flow.fetch_token(authorization_response=str(request.url))
    credentials = flow.credentials
    request_session = requests.Request()
    id_info = id_token.verify_oauth2_token(credentials._id_token, request_session, CLIENT_ID)
    user_email = id_info["email"]
    return RedirectResponse("/")  # Redirect back to index after login

# ---------- GCS File Upload ----------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        client = storage.Client()
        bucket = client.bucket('data-gpt')
        blob = bucket.blob(f"uploads/{file.filename}")

        contents = await file.read()

        blob.upload_from_string(contents, content_type=file.content_type)

        return HTMLResponse(f"<h2>✅ Uploaded {file.filename} successfully to GCS!</h2><a href='/'>Back</a>")

    except Exception as e:
        return HTMLResponse(f"<h2>❌ Upload failed: {e}</h2><a href='/'>Back</a>")

