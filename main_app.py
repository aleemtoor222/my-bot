from flask import Flask, request, jsonify
from uuid import uuid4
import logging
import base64
from langgraph.graph import StateGraph
from datetime import date, datetime
from pydantic import BaseModel, ValidationError, Field, field_validator
import psycopg2, os, json, logging, re
from langchain.document_loaders import PyPDFLoader
from typing import Optional, Dict, Any, List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import tempfile
from functools import lru_cache
from difflib import SequenceMatcher
from typing import Set
import jwt
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from azure.core.exceptions import ResourceExistsError
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.docstore.document import Document
import re, textwrap
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dataclasses import dataclass, field
import hashlib, json, os, pickle, shutil, tempfile, time
from functools import lru_cache
from typing import Any, Dict, List, Tuple
load_dotenv()
from collections import OrderedDict
from dateutil import parser as dateutil_parser 
from db_utils import conn
from llmconfig import AgentState
from tools import handle_user_input
from agent_graph import agent_executor
# Flask Setup
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
 
# Session Management (in-memory for demo; use Redis/DB in production)
session_store = {}
 
# Timeout configuration
timeout_seconds = 60
 
# ---------------------- LOGGING SETUP ----------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
 
# org_id = 2
 
# Azure Blob Storage configuration
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME","empower")
 

 

 


 
CANCEL_COMMANDS = {"cancel", "start over", "restart", "stop", "no", "nah", "nope", "nevermind"}
 
def detect_new_intent(user_input: str) -> Optional[str]:
    keywords = {
        "apply_leave": ["apply leave", "want leave", "take leave"],
        "apply_travel": ["apply travel", "travel request", "trip"],
        "apply_reimbursement": ["submit reimbursement", "expense claim"],
        "query_policy": ["policy", "guideline", "rule", "what is the policy", "show my reimbursements", "show my travel requests"]

    }
    user_input = user_input.lower()
    for fn, phrases in keywords.items():
        for phrase in phrases:
            if phrase in user_input or SequenceMatcher(None, phrase, user_input).ratio() > 0.85:
                return fn
    return None
 







 
def extract_employee_id_from_token(token: str) -> tuple[Optional[int], Optional[int], Optional[int], Optional[str], Optional[str]]:
    try:
        decoded = jwt.decode(token, os.getenv("JWT_SECRET", "ABC123DEF456GHI789JKLM101112NOP131415QRS161718TUV192021WXY222324Z2526"), algorithms=["HS256"])
        employeeId = decoded.get("employeeId")
        officeId = decoded.get("officeId")
        email = decoded.get("email")
        logger.debug(f"Token payload - employeeId: {employeeId}, officeId: {officeId}, email: {email}")
        if not isinstance(employeeId, int) or not isinstance(officeId, int):
            logger.error(f"Invalid types - employeeId: {type(employeeId)}, officeId: {type(officeId)}")
            return None, None, None, None, None
        with conn.cursor() as cur:
            # Fetch organizationId
            cur.execute('SELECT "organizationId" FROM "Office" WHERE id = %s', (officeId,))
            result = cur.fetchone()
            if not result:
                logger.error(f"No record found in Office table for officeId: {officeId}")
                return employeeId, None, officeId, None, None
            organizationId = result[0]
            # Fetch role
            cur.execute(
                'SELECT r.name FROM "Employee" e JOIN "Role" r ON e."roleId" = r.id WHERE e.id = %s AND e.is_deleted = FALSE',
                (employeeId,)
            )
            role_result = cur.fetchone()
            role = role_result[0].lower() if role_result else None
            logger.debug(f"organizationId: {organizationId}, role: {role}, email: {email}")
        return employeeId, organizationId, officeId, role, email
    except jwt.InvalidTokenError as e:
        logger.error(f"Token validation failed: {e}")
        return None, None, None, None, None
 
@app.route("/chat", methods=["POST"])
def chat():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth_header.split(" ")[1]
        global employeeId, org_id, officeId, email
        employeeId, org_id, officeId, role, email = extract_employee_id_from_token(token)
        logger.info(f"org_id: {org_id}, officeId: {officeId}, role: {role}")
        if employeeId is None:
            return jsonify({"error": "Invalid token or employeeId"}), 401
        if org_id is None:
            return jsonify({"error": "Invalid organization"}), 401
        if officeId is None:
            return jsonify({"error": "Invalid officeId in token"}), 401
        if role is None:
            return jsonify({"error": "Invalid role"}), 401
        if email is None:
            return jsonify({"error": "Invalid email in token"}), 401
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
                                    
        session_id = data.get("session_id")
        if not user_input:
            return jsonify({"error": "Missing 'user_input' field"}), 400
        if not session_id or session_id not in session_store:
            session_id = str(uuid4())
            state = AgentState(
                user_input=user_input,
                role=role,
                arguments={
                "email": email,
                "employeeId": employeeId,
                "org_id": org_id,
                "officeId": officeId
            }
            )
            logger.info(f"Created new session: {session_id}")
        else:
            state = session_store[session_id]
            state.user_input = user_input
            state.role = role
            state.timestamp = datetime.utcnow()
            if not state.arguments.get("email"):
                state.arguments["email"] = email
            if not state.arguments.get("employeeId"):
                state.arguments["employeeId"] = employeeId
            if not state.arguments.get("org_id"):
                state.arguments["org_id"] = org_id
            if not state.arguments.get("officeId"):
                state.arguments["officeId"] = officeId

        if state.wait_for_input and state.ask_user.startswith("Select an option"):
            state = handle_user_input(state, user_input)
        elif state.current_field:
            if user_input.lower() == "skip":
                state.arguments[state.current_field] = None
            else:
                state.arguments[state.current_field] = user_input
            state.current_field = None
            state.wait_for_input = False
            state.ask_user = ""
            state.user_input = ""

        # ── if the client has already picked a function, prime the state with it ──
        if data.get("function_name"):
            state.function_name   = data["function_name"]
            state.arguments       = data.get("arguments",    state.arguments)
            state.metadata        = data.get("metadata",     getattr(state, "metadata", {}))
            state.wait_for_input  = False
            state.current_field   = None
            state.validated       = {}

        result = agent_executor.invoke(state, config={"recursion_limit": 20})
        state = AgentState(**result)
        # state = run_agentic_session(state)
        # Update session store
        session_store[session_id] = state
        # Ensure response includes success message and follow-up prompt
        response_data = {
            "response": state.response,
            "ask_user": state.ask_user,
            "wait_for_input": state.wait_for_input,
            "session_id": session_id,
            "function_name": state.function_name,
            "arguments": state.arguments,
            "validated": state.validated,
            "tags":state.tags,
            "metadata": state.metadata

        }
        return jsonify(response_data)
    except Exception as e:
        logger.exception("Chat endpoint failed")
        return jsonify({"error": str(e)}), 500
 
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "HRMS Agent API is running"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8070)