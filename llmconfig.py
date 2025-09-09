from langchain.llms import LlamaCpp
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from azure.core.exceptions import ResourceExistsError
import os
import logging
import base64
from uuid import uuid4
from datetime import date, datetime
from dateutil.parser import parser
from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import Optional, Dict, Any, List, Set
import re
from langchain.document_loaders import PyPDFLoader
import hashlib
import pickle
import requests, tempfile
from dataclasses import dataclass, field
from langchain.docstore.document import Document
from xhtml2pdf import pisa
from io import BytesIO
from rapidfuzz import process, fuzz
# from langchain_community.llms import LlamaCppServer
from langchain_openai import ChatOpenAI



load_dotenv()
# Azure Blob Storage configuration
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME","empower")


# ---------------------- LOGGING SETUP ----------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LeaveForm(BaseModel):
    employeeId: int
    leave_type: str
    from_date: date
    to_date: date
    reason: str
 
class TravelRequestForm(BaseModel):
    employeeId: int
    travel_type: str
    city: str
    state: str
    country: str
    from_date: date
    to_date: date
    purpose_of_travel: str
    mode_of_transport: Optional[str] = None
 
class ItemizedExpense(BaseModel):
    type: str
    scannedImage: Optional[str] = None
    scannedAmount: int
 
    @field_validator("type")
    def strip_type(cls, v):
        return v.strip()
 
    @field_validator("scannedAmount")
    def validate_scanned_amount(cls, v):
        if v < 0:
            raise ValueError("scannedAmount must be non-negative")
        return v
 
class ReimbursementForm(BaseModel):
    employeeId: int
    reimbursement_type: str
    purpose_of_expense: str
    start_date: date
    end_date: date
    itemized_expense_type: Optional[str] = None
    itemized_expense_amount: Optional[float] = None
    itemized_expense_image: Optional[str] = None
    itemized_expenses: List[ItemizedExpense] = []
    total_amount: str
    currency: str
    image: Optional[str] = None
 
    @field_validator("image")
    def validate_image(cls, v):
        if v is None or v == "":
            return None
        # Check if it's a base64 string
        if v.startswith("data:image/"):
            try:
                # Extract base64 part
                base64_string = v.split(",")[1] if "," in v else v
                # Fix padding
                padding_needed = len(base64_string) % 4
                if padding_needed:
                    base64_string += '=' * (4 - padding_needed)
                base64.b64decode(base64_string)
                return v
            except Exception as e:
                raise ValueError(f"Invalid base64 image: {str(e)}")
        # Check if it's a URL
        url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
        if url_pattern.match(v):
            return v
        raise ValueError("Image must be a valid base64 string, URL, or null")
 
class AgentState(BaseModel):
    user_input: str
    function_name: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)
    validated: Dict[str, Any] = Field(default_factory=dict)
    response: str = ""
    ask_user: str = ""
    wait_for_input: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    current_field: Optional[str] = None
    already_asked: Set[str] = Field(default_factory=set)
    role: Optional[str] = None
    field_attempts: Dict[str, int] = Field(default_factory=dict)
    error_message: Optional[str] = None
    tags: Optional[Dict[str, Any]] = {}
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


 
class TeamAttendance(BaseModel):
    startDate: date
    endDate: date

    @field_validator("endDate")
    def check_date_range(cls, v, values):
        if "startDate" in values and v < values["startDate"]:
            raise ValueError("endDate must be on or after startDate")
        return v

class EditTimesheetForm(BaseModel):
    project_name: str
    timesheet_date: date
    clock_in: str
    clock_out: str

    @field_validator('clock_in', 'clock_out', mode='before')
    @classmethod
    def normalize_time(cls, v):
        if isinstance(v, str) and v.strip():
            
            norm = normalize_time_string(v)
            if not norm:
                raise ValueError("Invalid time format (use 09:00 or 2 PM)")
            return norm
        return v

    @field_validator('clock_out')
    @classmethod
    def check_time_logic(cls, v, info):
        clock_in = info.data.get("clock_in")
        if clock_in and v:
            fmt = "%H:%M"
            in_dt = datetime.strptime(clock_in, fmt)
            out_dt = datetime.strptime(v, fmt)
            if out_dt <= in_dt:
                raise ValueError("Clock-out time must be after clock-in.")
        return v

class teamTimesheet(BaseModel):
    employee_name: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: Optional[str] = None
    view_sub_team: Optional[bool] = False

class teamLeaves(BaseModel):
    employee_name: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: Optional[str] = None
class team(BaseModel):
    employee_name: Optional[str] = None



class clock_in(BaseModel):
    project_name: str
    geo_location: str

from dateutil import parser

def normalize_time_string(value: str) -> str | None:
    if not value:
        return None

    try:
        dt = parser.parse(value, fuzzy=True)
        return dt.strftime("%H:%M")  # Or "%H:%M:%S" if you need seconds
    except Exception:
        return None





@dataclass(slots=True)
class Cfg:
    index_root:    str = "faiss_employee_indexes"   # one sub-dir per employee
    index_k:       int  = 5                         # top-k hits
    policy_limit:  int  = 1_000                     # first N chars per PDF
    lru_size:      int  = 16                        # cached FAISS objs
    cache_ttl:     int  = 900                       # sec – payload cache
    similarity_threshold: float = 0.3
    # sentence templates (override freely)
    tpl: Dict[str,str] = field(default_factory=lambda: {
        "leave":         "Leave {leave_id} ({leave_type}) {from_date}→{to_date}, {days_count}d, status={status}",
        "leave_balance": "{leave_type} {year}: {remaining_leaves}/{allocated_leaves} left (used {used_leaves})",
        "travel":        "Travel {travel_id} ({travel_type}) {city}/{country} {from_date}→{to_date}, status={status}",
        "reimbursement": "Reimb {reimbursement_id} {reimbursement_type} {total_amount} {currency}, status={status}",
        "team_leave":    "Team {name} leave {leave_id} {leave_type} {from}→{to}, status={status}",
    })

CFG = Cfg()


class PolicyDocumentLoader:
    def __init__(self):
        self.cache_dir = "policy_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.local_pdf_path = os.path.join("data", "EmpowerX User Document.pdf")
 
    def load_policy(self, url: str) -> List[Document]:
        """
        Load organization-specific policy document from a URL and the static EmpowerX User Document.pdf.
        Returns a combined list of documents from both sources.
        """
        all_documents = []
 
        # Load organization-specific policy document from URL
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
       
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                all_documents.extend(pickle.load(f))
        else:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    tmp.write(response.content)
                    tmp.flush()
                   
                    loader = PyPDFLoader(tmp.name)
                    documents = loader.load()
                   
                    # Add metadata to identify as policy document
                    for doc in documents:
                        doc.metadata.update({
                            "policy_type": "policy",
                            "source": "url"
                        })
                   
                    with open(cache_file, "wb") as f:
                        pickle.dump(documents, f)
                   
                    all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to load policy document from {url}: {e}")
                raise
            finally:
                if 'tmp' in locals() and os.path.exists(tmp.name):
                    os.unlink(tmp.name)
 
        # Load static EmpowerX User Document.pdf
        cache_key = hashlib.md5(self.local_pdf_path.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
       
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                all_documents.extend(pickle.load(f))
        else:
            try:
                if not os.path.exists(self.local_pdf_path):
                    raise FileNotFoundError(f"Local PDF not found at {self.local_pdf_path}")
               
                loader = PyPDFLoader(self.local_pdf_path)
                documents = loader.load()
               
                # Add metadata to identify as user guide
                for doc in documents:
                    doc.metadata.update({
                        "policy_title": "EmpowerX User Guide",
                        "policy_type": "user_guide",
                        "source": "local_pdf"
                    })
               
                with open(cache_file, "wb") as f:
                    pickle.dump(documents, f)
               
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to load EmpowerX User Document from {self.local_pdf_path}: {e}")
                logger.warning("Proceeding without user guide documents.")
 
        return all_documents
policy_loader = PolicyDocumentLoader()


# try:


#     llm = LlamaCppServer(
#         model_url=os.getenv("MODEL_ENDPOINT", "74.162.67.10:8000"),
#         n_ctx=3072,
#         n_batch=128,
#         max_tokens=300,
#         temperature=0.4,
#     )

#     logger.debug("LLM initialized successfully")
# except Exception as e:
#     logger.error(f"LLM initialization failed: {e}")
#     raise

try:
    # llm = ChatOpenAI(
    #     model_path=os.getenv("LLAMA_MODEL_PATH", "./models/Phi-3.1-mini-128k-instruct-Q4_K_M.gguf"),
    #     lib_path="./llama.cpp/build/bin/Release/llama.dll",
    #     use_mmap=False,
    #     n_gpu_layers=33,
    #     n_threads=max(4, os.cpu_count() // 2),
    #     repeat_penalty=1.1,
    #     n_ctx=3072,            # plenty for a router; don’t chase 128k here
    #     n_batch=128,           # keep <=512 and <= n_ctx
    #     # --- decoding caps (routers don’t need long outputs) ---
    #     max_tokens=256,
    #     temperature=0.4,
    # )
    llm = ChatOpenAI(
    model="Phi-3.1-mini-128k-instruct-Q4_K_M.gguf",
    api_key="test-key", 
    base_url="http://74.162.67.10:8000/v1",
    temperature=0.4,
    max_tokens=256,

)

    logger.debug("LLM initialized successfully")
except Exception as e:
    logger.error(f"LLM initialization failed: {e}")
    raise

# from langchain_openai import ChatOpenAI
# import os, logging

# try:
#     llm = ChatOpenAI(
#         base_url=os.getenv("OPENAI_API_BASE", "http://llama-server:8000/v1"),
#         api_key=os.getenv("OPENAI_API_KEY", "dummy"),
#         model=os.getenv("OPENAI_MODEL", "phi-3.1-mini"),
#         temperature=0.4,
#         max_tokens=300,
#     )
#     logging.debug("ChatOpenAI (llama.cpp) initialized")
# except Exception as e:
#     logging.error(f"LLM init failed: {e}")
#     raise

# from langchain.llms import HuggingFaceEndpoint

# llm = HuggingFaceEndpoint(
#     endpoint_url=os.getenv("MODEL_ENDPOINT", "74.162.67.10:8000")
# )

def upload_base64_to_blob(base64_string: str, blob_name: str) -> str:
    """
    Upload a base64-encoded image to Azure Blob Storage and return the public URL.
    """
    try:
        if base64_string.startswith("data:image/"):
            base64_string = base64_string.split(",")[1]
        # Fix padding by adding '=' characters
        padding_needed = len(base64_string) % 4
        if padding_needed:
            base64_string += '=' * (4 - padding_needed)
        image_data = base64.b64decode(base64_string)
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        try:
            container_client.create_container()
        except ResourceExistsError:
            pass
        blob_path = f"images/reimbursement/{blob_name}_{uuid4()}.png"
        blob_client = container_client.get_blob_client(blob_path)
        blob_client.upload_blob(image_data, overwrite=True, content_type="image/png")
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{blob_path}"
        logger.debug(f"Uploaded image to Blob Storage: {blob_url}")
        return blob_url
    except Exception as e:
        logger.error(f"Failed to upload image to Blob Storage: {str(e)}")
        raise ValueError(f"Failed to upload image: {str(e)}")
    


def generate_pdf_and_upload_to_blob(html_content: str, pdf_name_prefix: str) -> str:
    try:
        # Generate PDF from HTML using xhtml2pdf
        pdf_stream = BytesIO()
        result = pisa.CreatePDF(src=html_content, dest=pdf_stream)
        if result.err:
            raise ValueError("Error during PDF generation")

        # Upload to Azure Blob
        AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not AZURE_CONNECTION_STRING:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set.")
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        try:
            container_client.create_container()
        except ResourceExistsError:
            pass

        blob_path = f"pdfs/timesheet/{pdf_name_prefix}_{uuid4()}.pdf"
        blob_client = container_client.get_blob_client(blob_path)
        pdf_stream.seek(0)
        blob_client.upload_blob(pdf_stream, overwrite=True, content_type="application/pdf")

        return f"https://{blob_service_client.account_name}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{blob_path}"
    except Exception as e:
        logging.error(f"PDF upload failed: {e}")
        raise ValueError("Failed to generate or upload PDF.")


def match_employee_names(user_input, employee_names, fuzzy_threshold=95):
    logger.debug(f"[NameExtractor] Matching names from: {user_input}")
    matched = []
    user_tokens = user_input.lower().split()

    for name in employee_names:
        name_lower = name.lower()

        # Exact match on any token
        if any(token == name_lower or token in name_lower for token in user_tokens):
            matched.append(name)
            continue

        # Fuzzy match only if VERY close
        score = fuzz.partial_ratio(user_input.lower(), name_lower)
        if score >= fuzzy_threshold:
            matched.append(name)

    matched = list(set(matched))  # Remove duplicates
    logger.debug(f"[NameExtractor] Final matched names: {matched}")
    return matched
