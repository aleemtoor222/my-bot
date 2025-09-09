from flask import Flask, request, jsonify
from uuid import uuid4
import logging
import base64
from langgraph.graph import StateGraph


from datetime import date, datetime, timedelta
from pydantic import BaseModel, ValidationError, Field, field_validator
import psycopg2, os, json, logging, re
from langchain.document_loaders import PyPDFLoader
from typing import Optional, Dict, Any, List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import requests
import tempfile
from functools import lru_cache
from difflib import SequenceMatcher
from typing import Set
import jwt
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from llmconfig import (llm, upload_base64_to_blob)
import numpy as np
from langchain.docstore.document import Document
import re, textwrap
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dataclasses import dataclass, field
import json, os, pickle, shutil, tempfile, time
from llm_router import detect_function_calls, format_multi_tool_response
from tools import *
from pdf_utils import query_policy
# from validation_utils import validate_edit_timesheet
from llmconfig import *
from db_utils import conn , send_push_notification, get_employee_names_for_org
from db_utils import is_clock_in_allowed
from typing import Any, Dict, List
from google.oauth2 import service_account
import google.auth.transport.requests
load_dotenv() 

from dateutil import parser as dateutil_parser 

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

global employeeId, org_id, officeId, email

SERVICE_ACCOUNT_FILE = './firebase_accountkey.json'
SCOPES = ['https://www.googleapis.com/auth/firebase.messaging']
PROJECT_ID = 'hn-employee-hrms-app'

FUNCTION_INFO = {
    "apply_leave": (LeaveForm, ["employeeId", "leave_type", "from_date", "to_date", "reason"]),
    "apply_travel": (TravelRequestForm, ["employeeId", "travel_type", "city","state", "country", "from_date", "to_date", "purpose_of_travel"]),
    "apply_reimbursement": (ReimbursementForm, ["employeeId", "reimbursement_type", "purpose_of_expense", "start_date", "end_date", "itemized_expense_type", "itemized_expense_amount","itemized_expense_image","itemized_expenses", "total_amount", "currency", "image"]),
    "get_team_attendance": (TeamAttendance, ["start_date", "end_date"]),
    "clock_in_to_project": (None, ["project_name" ,"geo_location"]),
    "personal_timesheet": (None, ["project_name", "start_date", "end_date", "status"]),
    "edit_timesheet_entry": (EditTimesheetForm, ["project_name","timesheet_date","clock_in","clock_out"]),
    "view_team_timesheet":(teamTimesheet, ["project_name", "start_date", "end_date", "employee_name", "status", "view_sub_team"]),
    "update_team_timesheet": (None, ["entries","start_date","end_date","employee_name","project_name","status","filter_status","reason"]),
    "update_timesheet_entry": (None, []),
    "clock_out_from_project": (None, []),
    "query_policy": (None, ["query"]),
    "show_team": (team, ["employee_name"]),
    # "show_team_leaves": (None, []),
    "team_leaves": (teamLeaves, ["employee_name","start_date" ,"end_date","status"]),
    "show_travel_requests": (None, []),
    "get_attendance": (None, []),
    "get_manager_detail": (None, []),
    "list_assigned_projects":  (None, []),

   
}

# def get_fcm_access_token():
#     credentials = service_account.Credentials.from_service_account_file(
#         SERVICE_ACCOUNT_FILE, scopes=SCOPES
#     )
#     request = google.auth.transport.requests.Request()
#     credentials.refresh(request)
#     return credentials.token

# def send_push_notification(fcm_token, title, body):
#     access_token = get_fcm_access_token()
#     logger.info("accessToken:{access_token}")
#     url = f'https://fcm.googleapis.com/v1/projects/hn-employee-hrms-app/messages:send'

#     headers = {
#         'Authorization': f'Bearer {access_token}',
#         'Content-Type': 'application/json'
#     }

#     payload = {
#         "message": {
#             "token": fcm_token,
#             "notification": {
#                 "title": title,
#                 "body": body
#             },
#             "data": {
#                 "title": title,
#                 "body": body
#             }
#         }
#     }

#     response = requests.post(url, headers=headers, json=payload)
#     if response.status_code == 200:
#         print("Notification sent successfully")
#         return True
    
#     else:
#         print(f"Failed to send notification: {response.text}")
#         return False
 
# ---------------------- DEEPSEEK / FUNCTION CALLING ----------------------
def clean_json_output(raw: str) -> str:
    raw = re.sub(r"<.*?>", "", raw).replace('\n', ' ').strip()
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    return match.group(0) if match else ""
 


intent_texts = {
    
    "apply_leave": [
        "I need leave", "apply for vacation", "take a day off ", "apply leave","apply my leeve","apkidk leave","please give me leave","appllly leeeve","aplly leve","apply mo leave"
    ],
    "apply_travel": [
        "I need to travel", "book me a business trip", "request travel","i need to travel","i wnat travel","i want travel request",'apply travel req'
    ],
    "apply_reimbursement": [
        "I want to submit an expense claim", "file reimbursement", "apple reimbursement" , "apply for reimbursement" ,"apply my reimbursement","i need reimbursement"
        "I want to apply for reimbursement" ,"I need reimbursement"
    ],
    "query_policy": [
        "my remaining leave balance",
        "what is my leave balance",
        "my leave balance",
        "show me my leaves",
        "show my leaves",
        "what is my leaves status"
        "show me my leave balance",
        "leave policy",
        "tell me about",
        "reimbursement policy",
        "travel policy",
        "what is casual leave",
        "what is casual leave policy",
        "what is the company policy",
        "tell me the guidelines",
        "show me the rules",
        "show my reimbursements",
        "list my expense claims",
        "my reimbursement details",
        "show me my reimbursements",
        "show my expense details",
        "show my reimbursement details"
        "view my expense reports",
        "show my travel requests",
        "list my trips",
        "phone number of",
        "contact details of",
        "give me the phone number",
        "employee contact",
        "my travel details",
        "show me my travel requests",
        "view my travel history",
        "travel requests for me",
        "my travel details",
        "who is [employee name]",
        "who is ayesha"
        "tell me about [employee name]",
        "employee details",
        "show employee information",
        "how can i apply",
        "how can i apply leave",
        "how to apply leave"
        "employee profile",
        "how can i apply",
        "how can i do that"
        "how my app will run",
        "how to aplly travel request",
        "how to apply this",
        "how to apply reimbursement request",
        "how to apply travel request",
        "how can i reimbursement",
        "how can i travel request"
        "how can I apply leave through app",
        "how can I use this app", "how to signup", "how to sign up",
        "how to login", "how to log in", "how can I add employee",
        "add an employee", "project overview", "what is this app",
        "how does  work", "app guide", "user guide",
        "how to get started", " tutorial",
        "signup process", "login process", "app usage instructions",
        "how to register employee", "employee registration process"
    ],
    "show_team": [
        "show me my team", "list my team members", "who is in my team",
        "my team details", "my team leaves"
    ],

    "clock_in_to_project":[
        "add my work log", "clock in for the project", "check in for [project_name]","start my work in the project","clock in to [project_name]","need to clock in", "clojck in"
    ],
    
    "clock_out_from_project":[
        "want to clock out","want to check out from project","clock out from project","clock out"
    ],

    "personal_timesheet":[
        "i wnat to see my timesheet","show my time sheet","My timesheet for yesterday","Show clock-in/out records for this week",'View timesheet for 25 June'
    ],

    "edit_timesheet_entry":[
        "edit my time sheet","Change my timesheet on 3 July. Clock in at 10am and out at 6pm","Set 3 July to 8:00 to 16:00","Edit 4 July clock-in to 9"
    ],
    
    "view_team_timesheet":[
        "show my team timesheet" , "my team timesheet" ,"team timesheet" , "show me ayesha timesheet"
    ],

   
    "team_leaves": [
        "show team leaves",
        "show [employee name]'s leaves",
        "[employee name] leave details",
        "leave details for [employee name]",
        "show leaves for employee [employee name]",
        "show me [employee_name] leaves",
        "display [employee_name]'s leave history",
        "show pending leaves of [employee_name]", 
        "show me pending leaves of [employee_name]",
        "pending leaves for [employee_name]", 
        "show approved leaves of [employee_name]",
        "show rejected leaves of [employee_name]",
        "show casual leaves of [employee_name]", 
        "show medical leaves of [employee_name]",
        "show sick leaves of [employee_name]",
        "show vacation leaves of [employee_name]",
        "show pending casual leaves of [employee_name]", 
        "show pending medical leaves of [employee_name]",
        "show casual leaves for [employee_name]",
        "view [employee_name]'s casual leave",
        "display casual leaves of [employee_name]"
    ],
    "get_attendance": [
        "show my attendance",
        "show my check-in time",
        "show my checkout time",
        "my attendance details",
        "view my attendance",
        "check my attendance"
    ],
    "get_team_attendance": [
        "show team attendance",
        "team attendance details",
        "view team attendance",
        "check team attendance",
        "team check-in details",
        "show [employee name]'s check-in time",  
        "show [employee name]'s checkout time",  
        "check-in time of [employee name]", 
        "checkout time of [employee name]",  
        "give me checkin time of [employee_name] for [date range]",  
        "give me checkout time of [employee_name] for [date range]" 
    ],
    "get_manager_detail": [
        "show my manager",
        "who is my manager",
        "show me [employee_name]'s manager",
        "who is [employee_name]'s manager",
        "manager details for [employee_name]"
    ]
}


# Load a lightweight sentence‐transformer
embed_model = SentenceTransformer("thenlper/gte-small")
# Compute one embedding per intent (mean of its examples)
intent_embeddings = {
    intent: [embed_model.encode(text, convert_to_numpy=True) for text in texts]
    for intent, texts in intent_texts.items()
}

def route_intent_with_embedding(user_input: str, threshold: float = 0.7) -> Optional[str]:
    user_input_lower = user_input.lower()
    vec = embed_model.encode([user_input], convert_to_numpy=True)[0]

    best_intent = None
    best_score = -1

    for intent, emb_list in intent_embeddings.items():
        for emb in emb_list:
            score = np.dot(vec, emb) / (np.linalg.norm(vec) * np.linalg.norm(emb))
            if score > best_score:
                best_score = score
                best_intent = intent

    logger.debug(f"Intent match: {best_intent} with score: {best_score}")

    question_words = {"how", "why"}
    if any(user_input_lower.startswith(q) for q in question_words):
        logger.debug("Rerouting to query_policy due to question-style input")
        return "query_policy"

    if best_score >= threshold:
        if best_intent == "team_leaves":
            words = user_input_lower.split()
            potential_names = [w for w in words if w not in ["show", "me", "team", "leaves", "leave", "details"]]
            if not potential_names:
                logger.debug("No employee name detected, falling back to show_team")
                return "show_team"
        return best_intent

    logger.debug(f"No intent matched, best score {best_score} below threshold {threshold}")
    return None

now = datetime.now()
full_date = now.strftime("%A, %Y-%m-%d")

def deepseek_call(prompt: str,employeeId=None, email=None, org_id=None, officeId=None, role=None) -> dict:
    """
    Route user prompt to appropriate function based on intent and arguments.
    """
    role = role or "employee"
    emp = employeeId or 0
    email = email or ""
    employee_names = get_employee_names_for_org(conn, org_id)
    matched_names = match_employee_names(prompt, employee_names)

    extra_hint = ""
    if matched_names:
        names_str = ", ".join(matched_names)
        extra_hint = f"\nThe extracted employee_name(s) are: '[{names_str}]' " \
                    f"You MUST use exactly these value(s) in 'employee_name'."\
                    f"Include all the names provided, do not miss any names"
        logger.debug(f"[NameExtractor] Injecting matched names into prompt: {matched_names}")
    else:
        logger.debug("[NameExtractor] No name matched — letting LLM decide.")



    instruction = f"""You are an AI function router that strictly follows these rules:
        1. Select exactly one function from: apply_leave, apply_travel, apply_reimbursement, query_policy, show_team, team_leaves, get_attendance,
        get_team_attendance, get_manager_detail,greetings,clock_in_to_project, check_out_from_project, personal_timesheet, edit_timesheet_entry, view_team_timesheet,list_assigned_projects
        Understand the user query intelligently and then choose the relevant function according to the user query 
        2. Return ONLY a JSON object with "function_name" and "arguments" keys
        {extra_hint}
        3. For apply_travel:
        - Do NOT extract or infer any other fields (e.g., city, country, travel_type) from the input or context
        
        4. For apply_leave:
        - Do NOT extract or infer any other fields (e.g., leave_type, from_date, to_date, reason) from the input or context
        5. query_policy :
            - if the user query is not related to other functions return this function like how are you, how about my leaves or like what is the policies , how to do 
            - For user queries like , show me my leaves , show my reimbursment details, show my expense details, where user asking about himself or related to his details, 
            How can i apply leave, questions in which how or what is being asked,hello, hi, hola, how are you
        FOR list_assigned_projects:
         - Triggered for queries like:
            show me assigned projects
            assigned projects to me
            tell me about my assigned projects
        6. clock_in_to_project:
            - Triggered for queries like:
            - "Start work"
            - "Check me in"
            - "Add my work log"
            - "Clock in to [project name]"
            - "clock in [project_name]"
            - Extract only: project_name (if mentioned),
        clock_out_from_project:
        - Queries like:
        - "checkout from project"
        - "clock out" 
        - "clock out from project"
        When user wants to edit_timesheet_entry :
            - 'clock_in' and 'clock_out' (in 24-hour format, e.g. "09:00", "17:30")
            - Do not use 12-hour format like "5pm" — convert it to "17:00", "9am" - convert it to "09:00"
        if employee_name is in user query Never use personal_timesheet
        for personal or asking like show my timesheets use personal_timesheets
        For personal_timesheet:
            - Use ONLY when the user is referring to their own timesheet.
            - If the user query contains another person's name or employee_name, always use view_team_timesheet instead.
            Examples
                - "Show my timesheet for last month" → `personal_timesheet`
                - show my approved timesheet
                - show my rejected timesheet
                - show my timesheets
                - its for personal like "my" or user wants its own time sheet
                - status should be in string
                -"i want to see my timesheet"
                -"show my time sheet"
                -"My timesheet for yesterday",
                -"Show clock-in/out records for this week"
                -'View timesheet for 25 June',
                -"show my rejected timesheet",
                -"show my pending timesheet"
        edit_timesheet_entry:
            Examples
            - i want to edit timesheet of 26 july
                here you have "timesheet_date", remaining arguments are null
            - add my previouse clockin and clock out for date
        For view_team_timesheet:
            Only view_sub_team is true when employee's team is asked name's team , name team
            - If the user asks for someone's team (e.g., "show me Asim's team timesheet"), include: "view_sub_team": True
            - If the user asks only for that person's timesheet (e.g., "show me Asim timesheet"), include: "view_sub_team": False
            - Trigger if:
                - Query contains **employee_name**  OR
                - Query contains **"team"**, **"employee"**, **"member"** OR
                - User role is `manager` or `HR`
            - Examples:
                - asim team timesheet
                - daim team timesheet
                - show Asim's team pending timesheet →  (employee_name="Asim", status="pending", "view_sub_team": true)
                - "show Asim pending timesheet" → `view_team_timesheet` (employee_name="Asim", status="pending", "view_sub_team": false)
                - "pending timesheet of Jawad" → `view_team_timesheet` (employee_name="Jawad", status="pending","view_sub_team": false)
                - "team timesheet for last month" → `view_team_timesheet` (employee_name=null, status=null, "view_sub_team": false)
        update_team_timesheet:
                This function is to approve or reject the employees timesheet, could have employee name, or could mention with project name, or user could as for multiple employees, you can find arguments the
                could be found in user query below in given functions
                – If the user is giving an imperative command to **approve**, **reject** or **decline** entries, choose **update_team_timesheet**.
                Examples
                - If user says "approve pending timesheets", set:
                    "status" = "APPROVED"
                    "filter_status" = "PENDING"
                - If user says "reject pending timesheet for this week", set:
                    "status" = "REJECTED"
                    "filter_status" = "PENDING"
                -"approve pending timesheet of my team"
                    Automatically extract:

                        status = "APPROVED" (from "approve")

                        filter by status = "PENDING" (from "pending timesheet")

                        and understand that it applies to team members, employee_name will be null
                - "Approve timesheet for Ayesha"
                - "Reject timesheets for team for this week"
                - "Approve timesheets"
                - "Approve John’s timesheet today"
                - "Approve Ali Raja's timesheet today" 
                - "Reject timesheets for this week"
                - Resolve relative dates (e.g., "today", "yesterday", "this week") using today's date: {full_date}.
                - Never include future date or tommorow date. 
        7. functions and their possible arguments: Only allowed functions
            Undertstand the user query and extract the relevant arguments intelligently
        - apply_leave: leave_type, from_date, to_date, reason
        - apply_travel: travel_type, city, country, from_date, to_date, purpose_of_travel, mode_of_transport
        - apply_reimbursement:
        - query_policy: query
        - show_team: employee_name
        - team_leaves: employee_name,start_date ,end_date,status
        - get_attendance: email, employeeId
        - get_team_attendance: start_date, end_date, personName
        - get_manager_detail: employee_name
        - clock_in_to_project: project_name
        - clock_out_from_project: none
        - personal_timesheet:"project_name", start_date, end_date, "status"
        - edit_timesheet_entry: project_name, timesheet_date, clock_in, clock_out
        - view_team_timesheet: project_name, start_date, end_date, employee_name, "status", view_sub_team
        - update_team_timesheet: start_date, end_date ,"employee_name", "status", project_name,"filter_status", reason
        - list_assigned_projects:

        if any arguments not found in user query or confused, add:  null
        arguments value should be in "string"
        11. Use team_leaves only when talk about team leaves or could ask about like show employee_name team leaves then according to employee_name should come leaves
            - show me [employee_name] leaves 
            - show me "employee_name" "status" leaves
        12. Query policy is for informational queries like leave balance, policies
        13. Use get_team_attendance for check-in/checkout queries with an employee name
        Today’s date is {full_date}   
        14. Date Handling:
        - start_date and end_date cannot be in future.
        - Convert expressions like "yesterday", "last week", "this week", "past 2 days", etc. into real date(s) based on today’s date.
        - Return dates in "YYYY-MM-DD" format.
        Examples:
        <JSON>{{"function_name":"apply_leave","arguments":{{"leave_type":"medical","from_date":"2025-05-01","to_date":"2025-05-02"}}}}</JSON>
        <JSON>{{"function_name":"apply_leave","arguments":{{}}}}</JSON>
        <JSON>{{"function_name":"apply_travel","arguments":{{"employeeId":{emp}}}}}</JSON>
        <JSON>{{"function_name":"query_policy","arguments":{{"query":"travel policy","employeeId":62}}}}</JSON>
        <JSON>{{"function_name":"show_team","arguments":{{"employee_name":null}}}}</JSON>
        <JSON>{{"function_name":"get_team_attendance","arguments":{{"personName":"","start_date":"","end_date":""}}}}</JSON>
        <JSON>{{"function_name":"team_leaves","arguments":{{"employee_name":null ,"start_date":null,"end_date":null,"status":null}}}}</JSON>
        - Do Not include "User query" and any "instructions" provided to you in your final response
        - Understand the context of user query
        - Do not invent any information or anything other then what is asked.
        - Return ONLY a JSON object with "function_name" and "arguments" keys according to the user query.Wrap it in <JSON></JSON> tags.
        User role: {role}
        Now process this input:
        User query: {prompt}
        Output:"""


    raw = llm.invoke(instruction, temperature=0.5)
    logger.debug(f"[deepseek] raw:\n{raw!r}")
 
    # Check embedding-based intent
    intent = route_intent_with_embedding(prompt)
    
    input_vec = embed_model.encode([prompt], convert_to_numpy=True)[0]
 
    sims = {
    intent: max(
        np.dot(input_vec, emb_vec) / (np.linalg.norm(input_vec) * np.linalg.norm(emb_vec))
        for emb_vec in emb_list
    )
    for intent, emb_list in intent_embeddings.items()
    }
    best_intent, best_score = max(sims.items(), key=lambda kv: kv[1])
 
    # Check for check-in/checkout with employee name (priority for get_team_attendance)
    name_match = re.search(r"(?:check-in\s*time|checkout\s*time|checkin\s*time|check\s*in\s*time|check\s*out|attendance)\s*(?:of\s*|for\s+)(?!my\b)([\w\s]+?)(?=\s*(?:on|for|from|\d{4}-\d{2}-\d{2}|$))", prompt, re.IGNORECASE)
    if name_match:
        logger.debug(f"[deepseek] Detected check-in/checkout with name: {name_match.group(1)} (score: {best_score})")
        args = {"personName": name_match.group(1).title().strip()}
       
        # Try to parse date from the prompt (e.g., "12 June" or "June 12")
        date_pattern = r"(?:(?:on|for)\s+)?(\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s*(?:\d{4})?)"
        date_match = re.search(date_pattern, prompt.lower())
        if date_match:
            try:
                # Parse natural language date (e.g., "12 June" or "12 June 2025")
                parsed_date = dateutil_parser.parse(date_match.group(1), dayfirst=True)
                args["start_date"] = parsed_date.strftime("%Y-%m-%d")
                args["end_date"] = parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                logger.error(f"[deepseek] Failed to parse date: {date_match.group(1)}")
                today = datetime.utcnow().strftime("%Y-%m-%d")
                args["start_date"] = today
                args["end_date"] = today
        else:
            # Fallback to current date if no valid date is found
            today = datetime.utcnow().strftime("%Y-%m-%d")
            args["start_date"] = today
            args["end_date"] = today
        return {"function_name": "get_team_attendance", "arguments": args}
 
    args = {"employeeId": emp}
    # if intent == "show_team":
    #     logger.debug(f"[deepseek] Embedding match for show_team (score: {best_score})")
    #     args = {"employeeId": emp}
 
    #     name_match = re.search(
    #         r"(?:show|view)(?:\s+me)?\s+(?!my\b)([\w\s]+?)(?:'s)?\s*team",
    #         prompt,
    #         re.IGNORECASE
    #     )
 
    #     if name_match:
    #         extracted_name = name_match.group(1).strip()
    #         normalized = extracted_name.lower()
    #         invalid_keywords = {"me", "my", "team", "me my", "my team", "show", "view"}
 
    #         if normalized not in invalid_keywords and not all(w in invalid_keywords for w in normalized.split()):
    #             args["employee_name"] = extracted_name.title()
    #             logger.debug(f"[deepseek] Extracted valid employee_name: {args['employee_name']}")
    #         else:
    #             logger.debug(f"[deepseek] Ignored extracted name due to being generic: {extracted_name}")
 
    #     return {"function_name": "show_team", "arguments": args}
    

    
    # Prioritize get_manager_detail for high confidence
    if intent == "get_manager_detail" and best_score >= 0.95:
        logger.debug(f"[deepseek] High-confidence embedding match for get_manager_detail (score: {best_score})")
        args = {"employeeId": emp}
        name_match = re.search(r"(?:show|view)(?:\s+me)?\s+(?!my\b)([\w\s]+?)(?:'s)?\s*manager", prompt, re.IGNORECASE)
        if name_match:
            args["employee_name"] = name_match.group(1).strip()
        return {"function_name": "get_manager_detail", "arguments": args}
 
    # Parse LLM response
    # match = re.search(r"<JSON>(.*?)</JSON>", raw, re.DOTALL)
    if hasattr(raw, "content"):
        raw = raw.content  # extract text from AIMessage

    match = re.search(r"<JSON>(.*?)</JSON>", raw, re.DOTALL)

    if match:
        cleaned = match.group(1).strip()
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict) and "function_name" in obj and isinstance(obj.get("arguments"), dict):
                # Override query_policy if embedding suggests get_manager_detail or show_team
                if obj["function_name"] == "query_policy" and intent in ["get_manager_detail"] and best_score >= 0.9:
                    logger.debug(f"[deepseek] Overriding query_policy with {intent} (score: {best_score})")
                    args = {"employeeId": emp}
                    if intent == "get_manager_detail":
                        name_match = re.search(r"(?:show|view)(?:\s+me)?\s+(?!my\b)([\w\s]+?)(?:'s)?\s*manager", prompt, re.IGNORECASE)
                    # if name_match:
                    #     args["employee_name"] = name_match.group(1).strip()
                    return {"function_name": intent, "arguments": args}
                if obj["function_name"] == "apply_travel":
                    obj["arguments"] = {"employeeId": emp}
                elif obj["function_name"] in ["get_manager_detail"]:
                    obj["arguments"]["employeeId"] = emp
                    name_match = re.search(r"(?:show|view)(?:\s+me)?\s+(?!my\b)([\w\s]+?)(?:'s)?\s*(?:manager|team)", prompt, re.IGNORECASE)
                    if name_match:
                        obj["arguments"]["employee_name"] = name_match.group(1).strip()
                elif obj["function_name"] == "get_attendance":
                    obj["arguments"]["email"] = email
                    obj["arguments"]["employeeId"] = emp
                elif obj["function_name"] == "query_policy":
                    obj["arguments"]["employeeId"] = emp
                elif obj["function_name"] == "get_team_attendance":
                    if not obj["arguments"].get("start_date") or not obj["arguments"].get("end_date"):
                        date_match = re.search(r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", prompt.lower())
                        if date_match:
                            obj["arguments"]["start_date"] = date_match.group(1)
                            obj["arguments"]["end_date"] = date_match.group(2)
                        else:
                            today = datetime.utcnow().strftime("%Y-%m-%d")
                            obj["arguments"]["start_date"] = today
                            obj["arguments"]["end_date"] = today
                    name_match = re.search(r"(?:check-in\s*time|checkout\s*time|checkin\s*time|check\s*in\s*time|check\s*out|attendance)\s*(?:of\s*|for\s+)(?!my\b)([\w\s]+?)(?=\s*(?:on|for|from|\d{4}-\d{2}-\d{2}|$))", prompt, re.IGNORECASE)
                    if name_match:
                        obj["arguments"]["personName"] = name_match.group(1).strip()
                                # ✅ Validate function_name and clean arguments
                fn = obj.get("function_name", "")
                args = obj.get("arguments", {})


                if fn in FUNCTION_INFO:
                    _, allowed_keys = FUNCTION_INFO[fn]
                    if not isinstance(args, dict):
                        args = {}
                    else:
                        args = {k: v for k, v in args.items() if k in allowed_keys}
                    obj["arguments"] = args
                    return obj
                else:
                    logger.warning(f"[deepseek] Unknown function name: {fn}")
                    return {"function_name": "", "arguments": {}}


        except json.JSONDecodeError as e:
            logger.error(f"[deepseek] JSON parse error: {str(e)}")
            return {"function_name": "error", "arguments": {"message": "Invalid JSON response"}}
 
    # Fallback to embedding-based intent
    
 
    # elif intent == "team_leaves":
    #       args = {"employeeId": emp}
    #       name_match = re.search(r"(?:show|view)(?:\s+me)?\s+(?!my\b)([\w\s]+?)(?:'s)?\s*(?:leaves|leave)", prompt, re.IGNORECASE)
    leave_type_match = None
    if name_match:
        args["employee_name"] = name_match.group(1).strip()
        leave_type_match = re.search(r"(casual|annual|sick|medical|vacation)", prompt, re.IGNORECASE)
    if leave_type_match:
        args["leave_type"] = leave_type_match.group(1).lower()
    if name_match:
        args["employee_name"] = name_match.group(1).strip()
    
    # Extract leave type (e.g., casual, annual, sick)
    leave_type_match = re.search(r"(casual|annual|sick|medical|vacation)", prompt, re.IGNORECASE)
    if leave_type_match:
        args["leave_type"] = leave_type_match.group(1).lower()
    
    # Extract status (e.g., pending, approved, rejected)
    status_match = re.search(r"(pending|approved|rejected)", prompt, re.IGNORECASE)
    if status_match:
        args["status"] = status_match.group(1).lower()
    elif intent == "get_manager_detail":
        args = {"employeeId": emp}
        name_match = re.search(r"(?:show|view)(?:\s+me)?\s+(?!my\b)([\w\s]+?)(?:'s)?\s*manager", prompt, re.IGNORECASE)
        if name_match:
            args["employee_name"] = name_match.group(1).strip()
    elif intent == "query_policy":
        args["query"] = prompt.strip()
    elif intent == "get_attendance":
        args["email"] = email
    elif intent == "get_team_attendance":
        name_match = re.search(r"(?:check-in\s*time|checkout\s*time|checkin\s*time|check\s*in\s*time|check\s*out|attendance)\s*(?:of\s*|for\s+)(?!my\b)([\w\s]+?)(?=\s*(?:on|for|from|\d{4}-\d{2}-\d{2}|$))", prompt, re.IGNORECASE)
        if name_match:
            args["personName"] = name_match.group(1).strip()
        date_match = re.search(r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", prompt.lower())
        if date_match:
            args["start_date"] = date_match.group(1)
            args["end_date"] = date_match.group(2)
        else:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            args["start_date"] = today
            args["end_date"] = today
    return {"function_name": intent, "arguments": args}

 

 
def validate_enum_field(state: AgentState, field_name: str, valid_values: List[str]) -> Optional[AgentState]:
    val = state.arguments.get(field_name)
    user_val = str(val).strip().lower() if val else ""
    if state.user_input.strip().lower() in CANCEL_COMMANDS:
        return reset_state(state)
    if user_val not in valid_values:
        if user_val in CANCEL_COMMANDS:
            return reset_state(state)
        state.response = (
            f"😕 I couldn't find '{user_val}' in the allowed options for {field_name.replace('_', ' ')}.\n"
            f"Here are the valid options: {', '.join(valid_values)}\n"
            f"If you want to cancel please press cancel"
        )
        state.ask_user = f"Please re-enter a valid {field_name.replace('_', ' ')}:"
        state.wait_for_input = True
        state.tags = {"show_cancel": True}
        state.current_field = field_name
        state.already_asked.discard(field_name)
        return state
    return None

 
def get_valid_leave_types(org_id: int) -> List[str]:
    with conn.cursor() as cur:
        cur.execute('SELECT type FROM "LeaveType" WHERE "organizationId" = %s', (org_id,))
        return [row[0].strip().lower() for row in cur.fetchall()]
 
def get_valid_travel_types(org_id: int) -> List[str]:
    with conn.cursor() as cur:
        cur.execute('SELECT type FROM "TravelType" WHERE "organizationId" = %s', (org_id,))
        return [row[0].strip().lower() for row in cur.fetchall()]
 
def get_valid_reimbursement_types(org_id: int) -> List[str]:
    with conn.cursor() as cur:
        cur.execute('SELECT type FROM "ReimbursementType" WHERE "organizationId" = %s', (org_id,))
        return [row[0].strip().lower() for row in cur.fetchall()]
 
CANCEL_COMMANDS = {"cancel", "start over", "restart", "stop"}
 
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
 
def reset_state(state: AgentState, message: str = "❌ Operation cancelled.") -> AgentState:
    args = state.arguments
    return AgentState(
        user_input="",
        response=message + " What would you like to do next?",
        role=state.role,
        arguments={"email": args.get("email"), "employeeId": args.get("employeeId"), "org_id": args.get("org_id"), "officeId": args.get("officeId")},
        wait_for_input=False,
        current_field=None,
        validated={},
        function_name="",
        already_asked=set()
    )



def parse_function(state: AgentState) -> AgentState:
    """
    1) Skip if we’re mid‐slot or already have intent
    2) Check for cancel/reset commands
    3) Call deepseek_call → route to clarify or collect_<fn>
    """




    # ✅ Top-level override: directly execute if function_name is already known and valid
    if state.function_name and state.function_name in FUNCTION_INFO:
        logger.debug(f"[parse_function] Skipping LLM for known function: {state.function_name}")
        return state

    employeeId = state.arguments.get("employeeId")
    email = state.arguments.get("email")
    role = state.role
    org_id = state.arguments.get("org_id")  # if you save this in state in main
    officeId = state.arguments.get("officeId")
    user_input = state.user_input.strip()
    logger.debug(f"parse_function input: {state}")


    if not state.wait_for_input:
        state.metadata = {}
        state.tags = {}

    if state.current_field or state.wait_for_input:
        return state

    if state.user_input.strip().lower() in CANCEL_COMMANDS:
        return reset_state(state)
    # 1) If we’re asking for a specific field, don’t re‐parse intent
    if state.function_name == "query_policy":
        current_query = state.arguments.get("query", "")
        if state.user_input.strip() != current_query.strip():
            state.arguments["query"] = state.user_input.strip()
        return state
    
    # 🔁 Map project_name → project_id if not already set
    if state.function_name == "clock_in_to_project":
        if "project_name" in state.arguments and not state.arguments.get("project_id"):
            try:
                name = state.arguments["project_name"].strip()
                emp_id = state.arguments.get("employeeId")

                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT p.id, p.name
                        FROM "Project" p
                        JOIN "EmployeeProject" ep ON ep."projectId" = p.id
                        WHERE ep."employeeId" = %s
                        AND p."isDeleted" = FALSE AND ep."isDeleted" = FALSE
                        AND p.name ILIKE %s
                        LIMIT 1
                    """, (emp_id, f"%{name}%"))

                    row = cur.fetchone()

                    if row:
                        state.arguments["project_id"] = row[0]
                        state.arguments.pop("project_name", None)
                        logger.debug(f"✅ Project matched: {row[1]} → ID {row[0]}")
                    else:
                        logger.warning(f"No assigned project matched for name: {name}")
                        state.function_name = ""
                        state.response = f"❌ You are not assigned to any project named like '{name}'. Please choose a valid project."
                        return state
            except Exception as e:
                logger.exception("Failed to map project name to ID")
                state.function_name = ""
                state.response = "❌ Error identifying project name. Please rephrase or try again."
                return state

       
    # If previous function is complete, allow re-parsing
    if (
        state.function_name in ["apply_leave", "apply_travel", "apply_reimbursement"]
        and not state.validated.get("pending_confirmation")
        and state.response.startswith(("Leave recorded", "Travel request created", "✅ Reimbursement logged"))
    ):
        logger.debug("parse_function: Previous function completed, resetting for new intent.")
        state.function_name = ""
        state.arguments = {
            "email": state.arguments.get("email"),
            "employeeId": state.arguments.get("employeeId")
        }
        state.validated = {}
        state.current_field = None
        state.already_asked = set()
    # Now skip re-parsing if function_name is still set (e.g., query_policy)
    if state.function_name:
        logger.debug("parse_function: function_name already set, skipping re-parse.")
        return state

    # 2) Cancel/reset?
    # if state.user_input.strip().lower() in CANCEL_COMMANDS:
    #     return AgentState(user_input="")
      # Reset if user cancels
    # ✅ This is fine
    

    # If previous function is complete, allow re-parsing
    if state.function_name and not state.validated.get("pending_confirmation"):
        # Check if the last response indicates a completed action
        if state.response.startswith(("Leave recorded", "Travel request created", "✅ Reimbursement logged")):
            logger.debug("parse_function: Previous function completed, allowing intent re-parsing.")
            state.function_name = ""
            state.arguments = {"email": state.arguments.get("email"), "employeeId": state.arguments.get("employeeId")}
            state.validated = {}
            state.already_asked = set()

    emb_intent = route_intent_with_embedding(state.user_input)



    if emb_intent == "query_policy":
        state.function_name = "query_policy"
        state.arguments = {
            "org_id": org_id,
            "office_id": officeId,
            "role": role,
            "employeeId": employeeId,
            "query": user_input
        }
        return state
    else:
    # 3) Ask LLM (with embed & keyword fallback) for intent+slots
        resp = deepseek_call(state.user_input,employeeId,email,org_id,officeId,role)

    # 4) If we still need clarification, show menu
    if resp["function_name"] == "clarify_intent":
        return clarify_intent(state)
    
    for key in ["employeeId", "email", "org_id", "officeId"]:
        if key not in resp["arguments"]:
            resp["arguments"][key] = state.arguments.get(key)

    # 5) Otherwise commit the intent and any extracted arguments
    valid_function_names = set(FUNCTION_INFO) | func_skip | {"clarify_intent"}
    fn = resp.get("function_name")

    if not fn or fn not in valid_function_names:
        logger.warning(f"parse_function: Invalid or unexpected function_name returned: {fn}")
        state.function_name = ""
        state.response = "❌ Sorry, I couldn't understand your request. Please rephrase or ask for help."
        state.wait_for_input = True  # ✅ Allow user to retry
        return state


    state.function_name = fn
    state.arguments = resp.get("arguments", {})

    logger.debug(f"parse_function output: {state}")
    return state
 
def clarify_intent(state: AgentState) -> AgentState:
    logger.debug("clarify_intent: Asking user to clarify intent.")

    state.response = (
        "🤖 I didn't quite catch that. But don't worry!\n"
        "Just type what you'd like to do — for example:\n"
        "• Apply for leave\n"
        "• Submit a reimbursement\n"
        "• Request a business trip\n"
        "• Ask about a policy or HR guideline\n\n"
        "I'm here to help!"
    )

    state.function_name = ""
    state.user_input = ""
    state.current_field = None
    state.wait_for_input = False
    state.ask_user = ""
    state.validated = {}
    state.error_message = None
    state.already_asked = set()
    state.field_attempts = {}
    return state
 
def collect_missing(state: AgentState, current_node_name: str = "") -> AgentState:
    args = state.arguments
    employeeId = args.get("employeeId")
    email = args.get("email")
    role = state.role
    org_id = args.get("org_id")  # if you save this in state in main
    officeId = args.get("officeId")
    logger.debug(f"collect_missing input: {state}")
    logger.debug(f"🧭 Inside collect_missing | Active Node: {current_node_name}, Function: {state.function_name}, Field: {state.current_field}")
    # ✅ This is fine
    if state.user_input.strip().lower() in CANCEL_COMMANDS:
        return reset_state(state)

    new_intent = detect_new_intent(state.user_input)
    if new_intent and new_intent != state.function_name:
        logger.debug(f"User expressed new intent: {new_intent}. Resetting state.")
        return AgentState(user_input=state.user_input)
    fn = state.function_name
    form_data = state.arguments
    _, required = FUNCTION_INFO.get(fn, (None, []))

    


    # if fn == "edit_timesheet_entry":
    #     clock_in_present = form_data.get("clock_in")
    #     clock_out_present = form_data.get("clock_out")
    #     if clock_in_present or clock_out_present:
    #         required = [f for f in required if f not in ("clock_in", "clock_out")]

 
    examples = {
        "leave_type": "Provided below Valid options",
        "from_date": "e.g., 2025-06-01 (YYYY-MM-DD)",
        "to_date": "e.g., 2025-06-05 (YYYY-MM-DD)",
        "reason": "e.g., family vacation",
        "travel_type": "e.g., business, training",
        "city": "e.g., Dubai",
        "state": "e.g New York",
        "country": "e.g., UAE",
        "purpose_of_travel": "e.g., client meeting",
        "reimbursement_type": "Provided below",
        "purpose_of_expense": "e.g., conference travel",
        "start_date": "e.g., 2025-06-01 (YYYY-MM-DD)",
        "end_date": "e.g., 2025-06-05 (YYYY-MM-DD)",
        "itemized_expense_type": "e.g., taxi, food, hotel",
        "itemized_expense_amount": "e.g., 120.50",
        "itemized_expense_image": "upload image",
        "total_amount": "e.g., 800",
        "image": "e.g., Your Expense Receipt or leave empty",
        "currency": "e.g., USD",
        "start_date":"e.g., 2025-06-01 (YYYY-MM-DD)",
        "end_date":"e.g., 2025-06-01 (YYYY-MM-DD)",
        "query": "e.g., leave encashment policy",
        "employee_name": "e.g., Ayesha or Ayeshax" ,
        "project_name": "e.g, Assigned projects",
        "timesheet_date":"",
        "clock_in":"e.g., 9:00",
        "clock_out":"e.g., 17:00",       
    }
    enum_fields = {
        "leave_type": get_valid_leave_types,
        "travel_type": get_valid_travel_types,
        "reimbursement_type": get_valid_reimbursement_types,
    }




    # Collect itemized expense from schema fields and loop if needed
    if state.function_name == "apply_reimbursement":
            
            if state.user_input.strip().lower() in CANCEL_COMMANDS:
                return reset_state(state)

            # 🛑 Handle response to "Would you like to add another itemized expense?" FIRST
            if state.current_field == "add_more_itemized":
                ans = state.user_input.strip().lower()
                if ans in {"yes", "y"}:
                    state.current_field = None
                    state.wait_for_input = True
                    return state
                elif ans in {"no", "n"}:
                    state.current_field = None
                    state.wait_for_input = False
                    logger.debug("User declined to add more itemized expenses")
                    state.arguments["itemized_expense_type"] = None
                    state.arguments["itemized_expense_amount"] = None
                    state.arguments["itemized_expense_image"] = None
                    return state  # 🛑 Ensure control flow exits here


                else:
                    state.ask_user = "Please respond with 'yes' or 'no'. Would you like to add another itemized expense?"
                    state.wait_for_input = True
                    return state    




            type_ = state.arguments.get("itemized_expense_type")
            amt = state.arguments.get("itemized_expense_amount")
            img = state.arguments.get("itemized_expense_image")

            if type_ and amt and img:
                # Upload if base64
                if img.startswith("data:image/"):
                    try:
                        blob = f"item_{type_.replace(' ', '_')}_{uuid4()}"
                        img = upload_base64_to_blob(img, blob)
                    except Exception as e:
                        state.ask_user = f"❌ Failed to upload image: {e}\nPlease re-upload:"
                        state.wait_for_input = True
                        return state

                # Append
                expense = {
                    "type": type_,
                    "scannedAmount": amt,
                    "scannedImage": img
                }
                if not isinstance(state.arguments.get("itemized_expenses"), list):
                    state.arguments["itemized_expenses"] = []
                state.arguments["itemized_expenses"].append(expense)

                # Reset
                state.arguments["itemized_expense_type"] = None
                state.arguments["itemized_expense_amount"] = None
                state.arguments["itemized_expense_image"] = None

                # Ask if more
                state.ask_user = "Would you like to add another itemized expense? (yes/no)"
                state.current_field = "add_more_itemized"
                state.wait_for_input = True
                return state


    for field in required:
        if state.user_input.strip().lower() in CANCEL_COMMANDS:
            return reset_state(state)
        if field in {"employeeId", "itemized_expenses"}:
            continue
        if state.arguments.get("add_more_itemized", "").strip().lower() == "no" and field in {
        "itemized_expense_type", "itemized_expense_amount", "itemized_expense_image"
         }:
            continue
        if not form_data.get(field) or (field == "reimbursement_type" and form_data.get(field) and not is_valid_reimbursement_type(form_data[field], org_id)):
            if field in state.already_asked and form_data.get(field):
                if state.current_field == field and state.wait_for_input:
                    if state.user_input.strip().lower() in CANCEL_COMMANDS:
                        return reset_state(state)
                    state.response = (
                        f"⚠️ The provided '{field.replace('_', ' ')}' is invalid.\n"
                        f"Please provide a valid value or type 'cancel' to leave it empty (if optional)."
                    )
                    state.ask_user = f"Please provide {field.replace('_', ' ')} ({examples[field]}):"
                    state.wait_for_input = True
                    state.tags = {"show_cancel": True}
                    state.already_asked.discard(field)  # Allow re-prompting
                    return state
                continue
            example = examples.get(field, "")
            extra_hint = ""
            if field in enum_fields:
                options = enum_fields[field](org_id)
                if options:
                    extra_hint = f"\nValid options: {', '.join(options)}"
            state.ask_user = f"Please provide {field.replace('_', ' ')} ({example}){extra_hint}:"
            state.wait_for_input = True
            state.current_field = field
            state.already_asked.add(field)
            logger.debug(f"collect_missing asking for: {field}, Value: {form_data.get(field)}, Already Asked: {state.already_asked}")
            return state
       




        # Validate image field immediately after input
        if field == "image" and form_data.get(field):
            try:
                ReimbursementForm.validate_image(form_data[field])
            except ValueError as e:
                state.response = f" Invalid image: {str(e)}. Please provide a valid URL or leave it empty."
                state.ask_user = f"Please provide {field.replace('_', ' ')} ({examples[field]}):"
                state.wait_for_input = True
                state.current_field = field
                state.already_asked.discard(field)
                return state
    logger.debug("collect_missing: All required fields collected")
    return state
 
def is_valid_reimbursement_type(reimbursement_type: str, org_id: int) -> bool:
    """Check if the reimbursement_type exists in the ReimbursementType table."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT id FROM "ReimbursementType" WHERE LOWER(type) = LOWER(%s) AND "organizationId" = %s',
                (reimbursement_type.strip(), org_id)
            )
            result = cur.fetchone()
            logger.debug(f"Validating reimbursement_type '{reimbursement_type}' for org_id {org_id}: {'Valid' if result else 'Invalid'}")
            return bool(result)
    except Exception as e:
        logger.error(f"Error validating reimbursement_type: {str(e)}")
        return False
def extract_field_update(user_input: str, valid_fields: List[str]) -> Optional[tuple]:
    user_input = user_input.lower()
    for field in valid_fields:
        field_words = field.replace('_', ' ')
        if field_words in user_input:
            match = re.search(rf"{re.escape(field_words)}.*?(to|is|=)\s+(.*)", user_input)
            if match:
                value = match.group(2).strip()
                return (field, value)
    return None
 
def validate(state: AgentState) -> AgentState:

    
    logger.debug(f"validate input: {state}")
    fn = state.function_name
    Model, _ = FUNCTION_INFO.get(fn, (None, []))

    args = state.arguments.copy()
    user_input = state.user_input.strip().lower()
    employeeId = args.get("employeeId")
    org_id = args.get("org_id")
    email = args.get("email")
    role = state.role
    officeId = args.get("officeId")



    # 🔹 1. Cancel handling
    if user_input in CANCEL_COMMANDS:
        return reset_state(state)

    # 🔹 2. Confirmation
    if state.validated.get("pending_confirmation"):
        if user_input in {"yes", "yep","yup"}:
            state.validated.pop("pending_confirmation")
            if fn == "edit_timesheet_entry":
                state.function_name = "update_timesheet_entry"
                state.metadata["from_edit_flow"] = True
            if state.function_name == "update_timesheet_entry":
                return update_timesheet_entry(state)
            state.ask_user = ""
            state.wait_for_input = False
            return state

        elif user_input in {"no", "nah", "nope", "nevermind"}:
            editable_fields = [k.replace('_', ' ') for k in state.validated if k != "pending_confirmation"]
            state.validated.pop("pending_confirmation", None)
            state.ask_user = "What would you like to change?"
            state.response = (
                f"No problem. You can say things like: 'change clock in to 9 AM'.\n"
                f"Editable fields: {', '.join(editable_fields)}"
            )
            state.wait_for_input = True
            return state

    # 🔹 3. Correction flow
    if not state.validated.get("pending_confirmation") and fn and state.ask_user.startswith("What would you like to change"):
        editable_fields = list(state.validated.keys())
        update = extract_field_update(state.user_input, editable_fields)
        if update:
            field, new_value = update
            args[field] = new_value.strip()
            state.arguments[field] = args[field]
            state.validated = {}
            try:
                validated = Model(**args)
                state.validated = validated.model_dump()
                state.validated["pending_confirmation"] = True
                summary = (
                    f"📝 I've updated `{field.replace('_', ' ')}` to: {args[field]}.\n\n"
                    f"Please confirm the updated {fn.replace('_', ' ')} details:\n"
                )
                for k, v in state.validated.items():
                    if k != "pending_confirmation":
                        summary += f"  {k.replace('_', ' ')}: {v}\n"
                summary += "Is this correct? (yes/no):"
                state.response = summary
                state.ask_user = summary
                state.wait_for_input = True
                return state
            except ValidationError:
                state.response = "❌ Error while updating. Please try again."
                state.wait_for_input = True
                return state
        else:
            state.response = "I couldn't understand which field to update."
            state.ask_user = "What would you like to change?"
            state.wait_for_input = True
            return state
    

    # 🔹 6. Handle remaining overlapping fields one-by-one
    if state.metadata.get("pending_overlaps"):
        next_field = state.metadata["pending_overlaps"].pop(0)
        value = state.arguments.get(next_field, "⏳")
        state.current_field = next_field
        state.ask_user = (
            f"⏰ Let's fix the next one.\nYour **{next_field.replace('_', ' ')}** time `{value}` also overlaps.\n"
            f"Please enter a different {next_field.replace('_', ' ')} time:"
        )
        state.wait_for_input = True
        return state



    # 🔹 4. Normalize times
    if fn == "edit_timesheet_entry":
        for field in ["clock_in", "clock_out"]:
            if field in args and args[field]:
                normalized = normalize_time_string(args[field])
                if normalized is None:
                    state.ask_user = f"⏰ Invalid time format for `{field}`. Please enter a valid time (e.g., 09:00 or 2 PM)."
                    state.current_field = field
                    state.wait_for_input = True
                    return state
                state.arguments[field] = normalized
                args[field] = normalized

    # 🔹 5. Normalize dates
    for date_field in ["from_date", "to_date", "start_date", "end_date", "date","timesheet_date"]:
        if date_field in args and isinstance(args[date_field], str):
            value = args[date_field].strip()
            if value.lower() in CANCEL_COMMANDS:
                return reset_state(state)
            try:
                parsed = dateutil_parser.parse(value)
                state.arguments[date_field] = parsed.date()
                args[date_field] = parsed.date()
            except ValueError:
                state.response = (
                    f"😕 It looks like the {date_field.replace('_', ' ')} is in the wrong format.\n"
                    f"Please enter it like this: `YYYY-MM-DD` (e.g., 2025-06-01).\n"
                    f"Or type CANCEL to skip."
                )
                state.ask_user = f"Could you re-enter the {date_field.replace('_', ' ')}?"
                state.wait_for_input = True
                state.tags = {"show_cancel": True}
                state.current_field = date_field
                state.already_asked.discard(date_field)
                return state
            
    # if fn == "edit_timesheet_entry":
    #     return validate_edit_timesheet(state)
    



    # Disallow editing today's date
    if fn in {"edit_timesheet_entry", "update_timesheet_entry"}:
        entry_date = state.arguments.get("date")
        if isinstance(entry_date, str):
            try:
                entry_date = dateutil_parser.parse(entry_date).date()
            except:
                pass
        if isinstance(entry_date, date):
            if entry_date == date.today():
                state.response = "❌ You can only edit entries from yesterday or earlier, not today."
                state.ask_user = "Please provide a different date:"
                state.current_field = "date"
                state.wait_for_input = True
                return state

    # 🔹 5. Overlap Check for Clock-In/Out edits
    if fn in {"edit_timesheet_entry", "update_timesheet_entry"} and "timesheet_edits" in state.metadata:
        edits = state.metadata["timesheet_edits"]
        for e in edits:
            clock_in = e.get("clock_in")
            clock_out = e.get("clock_out")
            entry_id = e.get("id")
            emp_id = state.arguments.get("employeeId")
            entry_date = e.get("date")
            project_name = state.arguments.get("project_name")

            if emp_id and entry_date and (clock_in or clock_out):
                try:
                    with conn.cursor() as cur:
                        project_id, _ = get_project_id_by_name(cur, emp_id, project_name)
                        entries = fetch_entries_for_date(cur, emp_id, None, entry_date)


                        def to_min(dt_str): return int(datetime.strptime(dt_str, "%H:%M").hour) * 60 + int(datetime.strptime(dt_str, "%H:%M").minute)

                        cur_ci_min = to_min(clock_in) if clock_in else None
                        cur_co_min = to_min(clock_out) if clock_out else None

                        overlapping_fields = []

                        for existing in entries:
                            if existing["id"] == entry_id:
                                continue
                            exist_ci = to_min(existing["start"])
                            exist_co = to_min(existing["end"])

                            if cur_ci_min and exist_ci <= cur_ci_min < exist_co:
                                overlapping_fields.append("clock_in")
                            if cur_co_min and exist_ci < cur_co_min <= exist_co:
                                overlapping_fields.append("clock_out")

                        # Remove duplicates
                        overlapping_fields = list(set(overlapping_fields))

                        if overlapping_fields:
                            field = overlapping_fields[0]
                            value = clock_in if field == "clock_in" else clock_out
                            state.current_field = field
                            state.ask_user = (
                                f"⏰ Your new **{field.replace('_', ' ')}** time `{value}` overlaps with another entry.\n"
                                f"Please enter a different {field.replace('_', ' ')} time (e.g., 10:30 AM):"
                            )
                            state.wait_for_input = True
                            # Save remaining ones to ask later
                            state.metadata["pending_overlaps"] = overlapping_fields[1:]
                            return state

                except Exception as e:
                    logger.warning(f"Overlap check failed: {e}")


    # 🔹 7. Handle multiple entries (A/B)
    if "pending_edit_entries" in state.metadata:
        index_map = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        reply = user_input.upper()
        if reply in index_map:
            idx = index_map.index(reply)
            matches = state.metadata["pending_edit_entries"]
            if idx < len(matches):
                selected = matches[idx]
                info = state.metadata["edit_info"]
                state.metadata["timesheet_edits"] = [{
                    "id": selected["id"],
                    "date": info["date"],
                    "clock_in": info["clock_in"],
                    "clock_out": info["clock_out"]
                }]
                state.validated["pending_confirmation"] = True
                new_clock_in = info.get("clock_in", "⏳ (unchanged)")
                new_clock_out = info.get("clock_out", "⏳ (unchanged)")

                state.ask_user = (
                    f"✅ Entry {selected['id']} selected.\n"
                    f"🕓 Old: Clock-in → {selected['start']}, Clock-out → {selected['end']}\n"
                    f"🆕 New: Clock-in → {new_clock_in}, Clock-out → {new_clock_out}\n\n"
                    f"Should I apply this update? (yes/no)"
                )

                state.wait_for_input = True
                return state


    # 🔹 6. Full edit_timesheet_entry execution
    if fn == "edit_timesheet_entry" and state.arguments.get("project_name") and state.arguments.get("timesheet_date") and state.arguments.get("clock_in") and  state.arguments.get("clock_out"):
        # if not state.arguments.get("clock_in") and not state.arguments.get("clock_out"):
        #     state.ask_user = "⏰ Please provide at least a new clock-in or clock-out time to proceed with editing."
        #     state.wait_for_input = True
        #     return state
        return edit_timesheet_entry(state)


    

    

    

    # 🔹 8. Reimbursement image handling
    if fn == "apply_reimbursement":
        image = args.get("image")
        if image and isinstance(image, str) and image.startswith("data:image/"):
            try:
                blob_name = f"receipt_{employeeId}"
                image_url = upload_base64_to_blob(image, blob_name)
                args["image"] = image_url
                state.arguments["image"] = image_url
            except Exception as e:
                state.response = f"❌ Failed to upload image: {e}"
                state.wait_for_input = True
                return state
        if not args.get("image"):
            state.response = "📸 Please upload your Manager approval image."
            state.ask_user = "Upload image:"
            state.wait_for_input = True
            state.current_field = "image"
            state.already_asked.discard("image")
            return state

    # 🔹 9. scannedImage URL check
    if "itemized_expenses" in args and isinstance(args["itemized_expenses"], list):
        for item in args["itemized_expenses"]:
            url = item.get("scannedImage")
            if url and not re.match(r'^https?://[^\s<>"]+', url):
                state.response = "Invalid scannedImage URL in itemized expenses."
                state.ask_user = "Please provide valid URLs for scannedImage:"
                state.wait_for_input = True
                state.current_field = "itemized_expenses"
                return state

    # 🔹 10. Enum validations
    enum_validations = {
        "leave_type": get_valid_leave_types(org_id),
        "travel_type": get_valid_travel_types(org_id),
        "reimbursement_type": get_valid_reimbursement_types(org_id)
    }
    for field, valid_values in enum_validations.items():
            if field in args:
                result = validate_enum_field(state, field, valid_values)
                if result:
                    return result
    # 🔹 11. Total amount = item sum check
    if "itemized_expenses" in args and "total_amount" in args:
        try:
            total = sum(item["scannedAmount"] for item in args["itemized_expenses"])
            if float(args["total_amount"]) != total:
                state.response = f"Total amount ({args['total_amount']}) does not match sum of itemized expenses ({total})."
                state.ask_user = "Please provide a correct total amount:"
                state.wait_for_input = True
                state.current_field = "total_amount"
                return state
        except Exception:
            state.response = "Invalid total amount."
            state.ask_user = "Please provide a valid total amount:"
            state.wait_for_input = True
            state.current_field = "total_amount"
            return state
        
    args.pop("itemized_expense_type", None)
    args.pop("itemized_expense_amount", None)
    args.pop("itemized_expense_image", None)

    # 🔹 12. Final pydantic validation + summary
    if Model:
        try:
            validated = Model(**args)
            state.validated = validated.model_dump()
            summary = f"Please confirm the following {fn.replace('_', ' ')} details:\n"
            for k, v in state.validated.items():
                if k == "itemized_expenses":
                    for item in v:
                        summary += f"  - {item['type']}: {item['scannedAmount']}\n"
                elif k == "image":
                    summary += f"  image: uploaded\n"
                elif k != "pending_confirmation":
                    summary += f"  {k.replace('_', ' ')}: {v}\n"
            summary += "Is this correct? (yes/no):"
            state.ask_user = summary
            state.wait_for_input = True
            state.validated["pending_confirmation"] = True
            return state
        except ValidationError as e:
            errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            state.response = ""
            state.error_message = f"Following are remaining Fields: {', '.join(errors)}"
            return state

    return state

def execute(state: AgentState) -> AgentState:
    logger.debug(f"execute input: {state}")
    fn = state.function_name
    data = state.validated
    cur = conn.cursor()
    args = state.arguments
    employeeId = state.arguments.get("employeeId")
    email = state.arguments.get("email")
    role = state.role
    org_id = state.arguments.get("org_id")  # if you save this in state in main
    officeId = state.arguments.get("officeId")
    try:
        if fn == "apply_leave":
            # Check if employee exists and fetch their name
            cur.execute(
                """SELECT id, name FROM "Employee"
                WHERE id = %s AND is_deleted = FALSE""",
                (data["employeeId"],)
            )
            emp = cur.fetchone()
            if not emp:
                state.response = f"Employee ID {data['employeeId']} not found."
                logger.debug(f"execute: Employee {data['employeeId']} not found")
                return state
            employee_name = emp[1]  

            # Check if leave type exists
            cur.execute(
                """SELECT id FROM "LeaveType"
                WHERE type ILIKE %s AND "organizationId" = %s AND is_deleted = FALSE""",
                (data["leave_type"], org_id)
            )
            lt = cur.fetchone()
            if not lt:
                state.response = f"Leave type {data['leave_type']} not found."
                logger.debug(f"execute: Leave type {data['leave_type']} not found")
                return state
            leave_type_id = lt[0]

            # Fetch working days for the organization
            cur.execute(
                """SELECT working_days
                FROM "Office"
                WHERE "organizationId" = %s AND is_deleted = FALSE""",
                (org_id,)
            )
            office = cur.fetchone()
            logger.info(f"officedetails:{office}")
            if not office:
                state.response = f"No office found for organization ID {org_id}."
                logger.debug(f"execute: Office not found for organization {org_id}")
                return state

            working_days = office[0]  

            # Map full day names to abbreviations for comparison
            day_name_map = {
                "Monday": "Mon",
                "Tuesday": "Tue",
                "Wednesday": "Wed",
                "Thursday": "Thu",
                "Friday": "Fri",
                "Saturday": "Sat",
                "Sunday": "Sun"
            }
            reverse_day_map = {v: k for k, v in day_name_map.items()}
            all_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

            # Identify weekend days (non-working days)
            weekend_days = [day for day in all_days if day not in working_days]

            # Calculate total leave days and working days
            total_days = (data["to_date"] - data["from_date"]).days + 1
            if total_days <= 0:
                state.response = "Invalid date range: to_date must be on or after from_date."
                logger.debug(f"execute: Invalid date range for employee {data['employeeId']}")
                return state

            # Collect all days in the leave period
            working_days_count = 0
            includes_non_working = False
            date_list = []
            current_date = data["from_date"]
            while current_date <= data["to_date"]:
                date_list.append(current_date)
                day_name = current_date.strftime("%A")
                abbreviated_day = day_name_map.get(day_name)
                if abbreviated_day in working_days:
                    working_days_count += 1
                else:
                    includes_non_working = True
                current_date += timedelta(days=1)

            # Check for sandwich leave
            is_sandwich = False
            if includes_non_working:
                # Find the last working day before the weekend and the first working day after
                last_working_before_weekend = None
                first_working_after_weekend = None
                for i in range(len(all_days)):
                    if all_days[i] in weekend_days:
                        # Look backward for the last working day
                        for j in range(i - 1, -1, -1):
                            if all_days[j] in working_days:
                                last_working_before_weekend = all_days[j]
                                break
                        # Look forward for the first working day
                        for j in range(i + 1, len(all_days)):
                            if all_days[j] in working_days:
                                first_working_after_weekend = all_days[j]
                                break
                        break  # Assume one continuous weekend block

                if last_working_before_weekend and first_working_after_weekend:
                    # Check if both surrounding working days are in the leave period
                    has_before = any(day_name_map.get(d.strftime("%A")) == last_working_before_weekend for d in date_list)
                    has_after = any(day_name_map.get(d.strftime("%A")) == first_working_after_weekend for d in date_list)
                    has_weekend = any(day_name_map.get(d.strftime("%A")) in weekend_days for d in date_list)

                    if has_weekend and has_before and has_after:
                        is_sandwich = True
                    else:
                        non_working_dates = [d.strftime('%Y-%m-%d') for d in date_list if day_name_map.get(d.strftime('%A')) not in working_days]
                        state.response = (
                            f"Leave period includes non-working days ({', '.join(non_working_dates)}). "
                            f"Non-working days can only be included if both the preceding ({reverse_day_map[last_working_before_weekend]}) "
                            f"and following ({reverse_day_map[first_working_after_weekend]}) working days are selected."
                        )
                        logger.debug(f"execute: Invalid leave period with non-working days for employee {data['employeeId']}")
                        return state

            if working_days_count == 0:
                state.response = "No working days included in the selected date range."
                logger.debug(f"execute: No working days in range for employee {data['employeeId']}")
                return state

            # Check for existing leave on the same dates
            cur.execute(
                """SELECT id FROM "Leave"
                WHERE "employeeId" = %s
                AND "fromDate" <= %s AND "toDate" >= %s
                AND is_deleted = FALSE""",
                (data["employeeId"], data["to_date"], data["from_date"])
            )
            existing_leave = cur.fetchone()
            if existing_leave:
                state.response = "You have already applied for a leave that overlaps with these dates."
                logger.debug(f"execute: Overlapping leave found for employee {data['employeeId']}")
                return state

            # Check leave balance
            current_year = datetime.utcnow().year
            cur.execute(
                """SELECT id, allocated_leaves, remaining_leaves, used_leaves
                FROM "LeavesBalance"
                WHERE "employeeId" = %s AND "leaveTypeId" = %s AND year = %s""",
                (data["employeeId"], leave_type_id, current_year)
            )
            balance = cur.fetchone()
            if not balance:
                state.response = f"No leave balance found for {data['leave_type']} in {current_year}."
                logger.debug(f"execute: No leave balance for employee {data['employeeId']}, leave type {leave_type_id}, year {current_year}")
                return state

            # Access tuple elements by index
            balance_id = balance[0]
            allocated_leaves = balance[1]
            remaining_leaves = balance[2]
            used_leaves = balance[3]

            if remaining_leaves < working_days_count:
                state.response = f"Insufficient {data['leave_type']} leave balance. Available: {remaining_leaves} days, Requested: {working_days_count} working days."
                logger.debug(f"execute: Insufficient leave balance for employee {data['employeeId']}")
                return state

            # Update leave balance based on working days
            new_remaining = remaining_leaves - working_days_count
            new_used = used_leaves + working_days_count
            cur.execute(
                """UPDATE "LeavesBalance"
                SET remaining_leaves = %s,
                    used_leaves = %s,
                    updated_at = %s
                WHERE id = %s""",
                (new_remaining, new_used, datetime.utcnow(), balance_id)
            )

            # Insert leave record with total days (including non-working days if sandwich)
            days_to_record = total_days if is_sandwich else working_days_count
            cur.execute(
                """INSERT INTO "Leave" ("employeeId", "leaveTypeId", "fromDate", "toDate", "reason", "username", "days_count", "created_at", "updated_at")
                VALUES (%s, %s, %s, %s, %s, %s, %s, now(), now())
                RETURNING id""",
                (data["employeeId"], leave_type_id, data["from_date"], data["to_date"], data["reason"], employee_name, days_to_record)
            )
            leave_id = cur.fetchone()[0]

            cur.execute(
                """SELECT m.id, m."fcmToken"
                FROM "Employee" e
                JOIN "Employee" m ON e."managerId" = m.id
                WHERE e.id = %s AND e.is_deleted = FALSE AND m.is_deleted = FALSE""",
                (data["employeeId"],)
            )
            manager = cur.fetchone()
            logger.info(f"managerdetails:{manager}")
            if manager:
                manager_id, fcm_token = manager

                send_push_notification(
                    fcm_token=fcm_token,
                    title="Leave Application",
                    body=f"{employee_name} has applied for {working_days_count} day(s) of {data['leave_type']} leave"
                )
                

            # Optionally store the notification in a table
            cur.execute(
                """INSERT INTO "Notification" ("senderId", "recipientId", "officeId", "notification_type", "title", "message", "status", "is_deleted", "created_at")
                VALUES (%s, %s, %s, %s, %s, %s, 'unread', FALSE, now())""",
                (
                    data["employeeId"],         # senderId
                    manager_id,                 # recipientId
                    data.get("officeId"),       # use appropriate officeId if available
                    "approval",                 # notification_type
                    "New Leave Request",
                    f"{employee_name} has applied for {working_days_count} day(s) of {data['leave_type']} leave"
                )
            )
            conn.commit()

            state.response = (
                f"Leave recorded for {days_to_record} days (including {working_days_count} working days). "
                f"Your remaining {data['leave_type']} balance is {new_remaining} days."
            )
            logger.debug(f"execute: Leave recorded for {days_to_record} days ({working_days_count} working days), new balance {new_remaining}, leave_id {leave_id}")
            return state
        elif fn == "apply_travel":
            cur.execute(
                """SELECT id, name FROM "Employee"
                WHERE id = %s AND is_deleted = FALSE""",
                (data["employeeId"],)
            )
            emp = cur.fetchone()
            if not emp:
                state.response = f" Employee with ID {data['employeeId']} not found."
                logger.debug(f"execute: Employee {data['employeeId']} not found")
                return state
            employee_name = emp[1]
            cur.execute("SELECT id FROM \"TravelType\" WHERE type ILIKE %s AND \"organizationId\" = %s", (data["travel_type"], org_id))
            tt = cur.fetchone()
            if not tt:
                state.response = f"Travel type {data['travel_type']} not found."
                logger.debug(f"execute: Travel type {data['travel_type']} not found")
                return state
            cur.execute( """
                INSERT INTO "TravelRequest" 
                ("employeeId", "travelTypeId", "city", "state", "country", "fromDate", "toDate",
                "purposeOfTravel", "modeOfTransport", "username", "created_at", "updated_at")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now())
                """,
                (
                data["employeeId"], tt[0], data["city"], data["state"], data["country"],
                data["from_date"], data["to_date"], data["purpose_of_travel"],
                data.get("mode_of_transport"), employee_name
                )
            )
            conn.commit()
            state.response = " Travel request created."
            logger.debug("execute: Travel request created")
        elif fn == "apply_reimbursement":
            cur.execute("SELECT id FROM \"Employee\" WHERE id=%s", (data["employeeId"],))
            emp = cur.fetchone()
            if not emp:
                state.response = f" Employee with ID {data['employeeId']} not found."
                logger.debug(f"execute: Employee {data['employeeId']} not found")
                return state
            cur.execute("SELECT id FROM \"ReimbursementType\" WHERE type ILIKE %s AND \"organizationId\" = %s", (data["reimbursement_type"], org_id))
            rt = cur.fetchone()
            if not rt:
                state.response = f" Reimbursement type '{data['reimbursement_type']}' not found for organization ID {org_id}."
                logger.debug(f"execute: Reimbursement type {data['reimbursement_type']} not found")
                return state
            cur.execute(
                """INSERT INTO "Reimbursement" ("employeeId", "reimbursementTypeId", "purposeofExpense",
                                                "start_date", "end_date", "itemizedExpenses", "totalamount", "currency", "image",
                                                "created_at", "updated_at")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now())""",
                (
                    data["employeeId"], rt[0], data["purpose_of_expense"],
                    data["start_date"], data["end_date"], json.dumps(data["itemized_expenses"]),
                    data["total_amount"], data["currency"], data["image"]
                )
            )
            conn.commit()
            state.response = "✅ Reimbursement logged."
            logger.debug("execute: Reimbursement logged")
        elif fn == "query_policy":
            state = query_policy(state)
            logger.debug("execute: Query policy executed")
        elif fn == "show_team":
            state = show_team(state)
        # elif fn == "show_team_leaves":
        #     state = show_team_leaves(state)
        elif fn == "team_leaves":
            state = team_leaves(state)
        elif fn == "get_attendance":
            state = get_attendance(state)  # Call the get_attendance function
            logger.debug("execute: Get attendance executed")
        elif fn == "get_team_attendance":
            state = get_team_attendance(state)
            logger.debug("execute: Get team attendance executed")
        elif fn == "get_manager_detail":
            state = get_manager_detail(state)
            logger.debug("execute: Get manager detail executed")
        elif fn == "clock_in_to_project":
            state = clock_in_to_project(state)
        elif fn == "clock_out_from_project":
            state = clock_out_from_project(state)
        elif fn == "personal_timesheet":
            state = personal_timesheet(state)
        elif fn == "update_timesheet_entry":
            state = update_timesheet_entry(state)
        elif fn == "view_team_timesheet":
            state = view_team_timesheet(state)
        elif fn == "list_assigned_projects":
            state = list_assigned_projects(state)
        elif fn == "greetings":
            state.response = "Hey! How can i assist you today"
            state.user_input = ""
        elif fn == "update_team_timesheet":
            state = update_team_timesheet(state)
        else:
            state.response = f" Unknown function: {fn}"
            logger.debug(f"execute: Unknown function {fn}")
    except Exception as e:
        conn.rollback()
        state.response = f" Database error: {str(e)}"

    finally:
        cur.close()
    logger.debug(f"execute output: {state}")
    return state



def handle_error(state: AgentState) -> AgentState:
    logger.error(f"handle_error: Invalid function triggered for input: {state.user_input}")
    state.wait_for_input = False

    return state


# ---------------------- GRAPH ASSEMBLY ----------------------
graph = StateGraph(state_schema=AgentState)
graph.add_node("parse", parse_function)
graph.add_node("validate", validate)
graph.add_node("execute", execute)
graph.add_node("respond", respond)
graph.add_node("clarify_intent", clarify_intent)
graph.add_node("show_team", show_team)
# graph.add_node("show_team_leaves", show_team_leaves)
graph.add_node("team_leaves", team_leaves)
graph.add_node("get_attendance", get_attendance)
graph.add_node("get_team_attendance", get_team_attendance)
graph.add_node("get_manager_detail", get_manager_detail)
graph.add_node("clock_in_to_project", clock_in_to_project)
graph.add_node("clock_out_from_project", clock_out_from_project)
graph.add_node("personal_timesheet",personal_timesheet)
graph.add_node("edit_timesheet_entry", edit_timesheet_entry)
graph.add_node("update_timesheet_entry", update_timesheet_entry)
graph.add_node("view_team_timesheet", view_team_timesheet)
graph.add_node("update_team_timesheet", update_team_timesheet)
graph.add_node("list_assigned_projects", update_team_timesheet)






graph.add_node("handle_error", handle_error)
for fn in FUNCTION_INFO:
    graph.add_node(f"collect_{fn}", collect_missing)
def log_condition(node, condition, state):
    result = condition(state)
    logger.debug(f"Conditional edge from {node}: {result}")
    return result
 
for fn in FUNCTION_INFO:
    graph.add_conditional_edges(
        f"collect_{fn}",
        lambda state: log_condition(f"collect_{fn}", lambda s: "ready" if all(s.arguments.get(field) is not None for field in FUNCTION_INFO[fn][1]) else "wait_for_input", state),
        path_map={"ready": "validate", "wait_for_input": f"collect_{fn}"}
    )


graph.add_conditional_edges(
    "validate",
    lambda state: log_condition("validate", lambda s:
        "validated"
        if (
            s.function_name == "query_policy" and "query" in s.arguments
        ) or (
            s.validated and not s.validated.get("pending_confirmation")
        ) else "response",
        state
    ),
    path_map={"validated": "execute", "response": "respond"}
)

graph.add_conditional_edges(
    "get_manager_detail",
    lambda state: log_condition("get_manager_detail", lambda s: "response" if not s.wait_for_input else "get_manager_detail", state),
    path_map={"response": "respond", "get_manager_detail": "get_manager_detail"}
)

graph.add_conditional_edges(
    "execute",
    lambda state: log_condition("execute", lambda s: "response", state),
    path_map={"response": "respond"}
)

graph.add_conditional_edges(
    "edit_timesheet_entry",
    lambda state: "execute" if state.function_name == "update_timesheet_entry" else "collect_edit_timesheet_entry",
    {
        "collect_edit_timesheet_entry": "collect_edit_timesheet_entry",
        "execute": "update_timesheet_entry",
    },
)



graph.add_conditional_edges(
    "get_attendance",
    lambda state: log_condition("get_attendance", lambda s: "response", state),
    path_map={"response": "respond"}
)
graph.add_conditional_edges(
    "get_team_attendance",
    lambda state: log_condition("get_team_attendance", lambda s: "response" if not s.wait_for_input else "get_team_attendance", state),
    path_map={"response": "respond", "get_team_attendance": "get_team_attendance"}
)
func_skip = {"query_policy", "show_team", "team_leaves", "get_attendance" ,"get_team_attendance" ,"clock_in_to_project","list_assigned_projects",
             "get_manager_detail","greetings","clock_out_from_project","personal_timesheet","update_timesheet_entry","view_team_timesheet","update_team_timesheet"}
graph.add_conditional_edges(
    "parse",
    lambda s: s.function_name or "clarify_intent",
    path_map={
        **{fn: f"execute" for fn in func_skip},
        **{fn: f"collect_{fn}" for fn in FUNCTION_INFO if fn not in func_skip},
        "clarify_intent": "clarify_intent",
        "error": "handle_error"

    },
)
graph.add_conditional_edges(
    "show_team",
    lambda state: log_condition("show_team", lambda s: "response", state),
    path_map={"response": "respond"}
)

# graph.add_conditional_edges(
#     "show_team_leaves",
#     lambda state: log_condition("show_team_leaves", lambda s: "response", state),
#     path_map={"response": "respond"}
# )

graph.add_conditional_edges(
    "team_leaves",
    lambda state: log_condition("team_leaves", lambda s: "response" if not s.wait_for_input else "team_leaves", state),
    path_map={"response": "respond", "team_leaves": "team_leaves"}
)





graph.set_entry_point("parse")
graph.set_finish_point("respond")
agent_executor = graph.compile()



