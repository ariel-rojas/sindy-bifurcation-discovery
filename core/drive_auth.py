#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drive Auth Module (Robust Version)
---------------------------------
- Usa OAuth2 (usuario)
- Reemplaza googleapiclient por requests (AuthorizedSession)
- Más estable en Windows y multithreading
"""

import os
import json
import socket
from dotenv import load_dotenv

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request, AuthorizedSession
import threading

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/drive']


# =============================================================================
# DEBUG UTIL
# =============================================================================

def debug(msg):
    print(f"[DEBUG] {msg}")


# =============================================================================
# NETWORK TEST
# =============================================================================

def test_basic_connectivity():
    try:
        host = "www.googleapis.com"
        ip = socket.gethostbyname(host)
        debug(f"DNS OK → {ip}")

        s = socket.create_connection((host, 443), timeout=5)
        s.close()
        debug("TCP OK (443)")
    except Exception as e:
        raise RuntimeError(f"Sin conectividad a Google APIs: {e}")


# =============================================================================
# AUTH CORE
# =============================================================================

thread_local = threading.local()

def get_drive_service():
    if hasattr(thread_local, "session"):
        return thread_local.session

    creds = None
    creds_path = os.getenv('DRIVE_CREDENTIALS_PATH')

    config_dir = os.path.dirname(os.path.abspath(creds_path))
    token_path = os.path.join(config_dir, 'token.json')

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, 'w') as f:
            f.write(creds.to_json())

    session = AuthorizedSession(creds)
    thread_local.session = session

    return session


# =============================================================================
# CONFIG
# =============================================================================

def get_target_folder_id():
    print("Obteniendo ID de carpeta de destino...")
    folder_id = os.getenv('DRIVE_TARGET_FOLDER_ID')
    if not folder_id:
        raise ValueError("DRIVE_TARGET_FOLDER_ID no definido en .env")
    return folder_id


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    try:
        print("Probando conexión...")
        s = get_drive_service()
        print("✅ OK")
    except Exception as e:
        print(f"❌ Error: {e}")