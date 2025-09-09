import hashlib
import pickle
import os, tempfile, requests, json, shutil, textwrap,time
from typing import Any, Dict, List, Tuple
from langchain.document_loaders import PyPDFLoader
from db_utils import (conn, get_employee_data)
from langchain.vectorstores import FAISS
from llmconfig import (CFG, policy_loader, AgentState, llm)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import OrderedDict
from functools import lru_cache
from langchain.chains.question_answering import load_qa_chain
from db_utils import build_employee_documents
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
 
# ---------------------- FAISS-POWERED POLICY BOT ----------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
 
cur = conn.cursor()
cur.execute("""SELECT title, document_url FROM "PolicyDocument" """)
docs = cur.fetchall()
logger.debug(f"Policy documents: {docs}")
 
all_docs = []
for title, url in docs:
    try:
        loaded = policy_loader.load_policy(url)
        logger.debug(f"Loaded {len(loaded)} pages from '{title}' (URL: {url})")
        all_docs.extend(loaded)
    except Exception as e:
        logger.error(f"Error loading {title} (URL: {url}): {e}")
 
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
split_docs = splitter.split_documents(all_docs)

if not split_docs:
    logger.warning("⚠️ No documents found — FAISS index not created.")
else:
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local("policy_faiss_index")
    logger.info("✅ FAISS index created and saved.")
 
vectordb = None
if os.path.exists("policy_faiss_index"):
    vectordb = FAISS.load_local("policy_faiss_index", embeddings, allow_dangerous_deserialization=True)
    logger.info("✅ FAISS index loaded.")
else:
    logger.warning("⚠️ No FAISS index to load. Skipping similarity search.")

qa_chain = load_qa_chain(llm, chain_type="stuff")
 
@lru_cache(maxsize=100)
def cached_similarity_search(query: str) -> tuple:
    if vectordb is None:
        logger.warning("⚠️ Similarity search requested but FAISS index is not available.")
        return tuple()
    docs = vectordb.similarity_search(query, k=10)
    return tuple([(doc.page_content, doc.metadata) for doc in docs])




os.makedirs(CFG.index_root, exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────────
#  Tiny helpers
# ────────────────────────────────────────────────────────────────────────────────
def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, default=str, sort_keys=True).encode()).hexdigest()

class _LRU(OrderedDict):
    def __init__(self, limit): 
        super().__init__() 
        self.limit = limit

    def get(self, k):
        item = super().get(k)
        if item:
            self.move_to_end(k)
        return item  # ✅ return full (value, timestamp) tuple

    def put(self, k, v):
        super().__setitem__(k, v)
        if len(self) > self.limit:
            self.pop(next(iter(self)))


FAISS_CACHE = _LRU(CFG.lru_size)
EMP_CACHE   = _LRU(64)



def fetch_employee(employee_id: int, org_id: int, office_id: int, role: str) -> Tuple[Dict[str, Any], str]:
    cached = EMP_CACHE.get(employee_id)
    if cached:
        payload, ts = cached
        if time.time() - ts < CFG.cache_ttl:
            h = _sha(payload)
            return payload, h
    payload = get_employee_data(employee_id, org_id, office_id, role)
    h = _sha(payload)
    EMP_CACHE.put(employee_id, (payload, time.time()))
    return payload, h

# ────────────────────────────────────────────────────────────────────────────────
#  3. build/refresh per-employee FAISS index  (hash marker test)
# ────────────────────────────────────────────────────────────────────────────────
def _idx_dir(eid:int)->str: return os.path.join(CFG.index_root,f"emp_{eid}")
def _load_index(eid:int)->FAISS:
    idx = FAISS_CACHE.get(eid)
    if idx:
        return idx
    path = _idx_dir(eid)
    if not os.path.exists(path):
        logger.warning(f"❌ Index path not found for employee {eid}")
        return None
    try:
        idx = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        FAISS_CACHE.put(eid, idx)
        return idx
    except Exception as e:
        logger.error(f"⚠️ Failed to load FAISS index for employee {eid}: {e}")
        return None


def ensure_index(emp: Dict[str, Any], h: str, org_id: int, office_id: int):
    path=_idx_dir(emp["employeeId"]); marker=os.path.join(path,".hash")
    if os.path.isfile(marker):
        try:
            with open(marker, "r") as f:
                saved_hash = f.read().strip()
            if saved_hash == h:
                return _load_index(emp["employeeId"])
        except Exception as e:
            logger.warning(f"Failed to read index marker: {e}")
    if os.path.isfile(marker) and open(marker).read().strip()==h:
        return _load_index(emp["employeeId"])
    # rebuild
    if os.path.exists(path): shutil.rmtree(path)
    logger.debug(f"Building documents for employee {emp['employeeId']}")
    docs = build_employee_documents(emp , org_id , office_id)
    if not docs:
        logger.warning(f"⚠️ No documents for employee {emp['employeeId']}. Skipping index creation.")
        return None

    
    logger.info(f"docs:{docs}")
    for p in emp["leave_policies"]:
        try:
            raw = policy_loader.load_policy(p["document_url"])[:CFG.policy_limit]
            docs.extend(splitter.split_documents(raw))
        except Exception as e: logger.warning(f"Policy {p['title']} skipped: {e}")
    if docs:
        idx = FAISS.from_documents(docs, embeddings)
        idx.save_local(path)
        open(marker,"w").write(h)
        FAISS_CACHE.put(emp["employeeId"], idx)
        return idx
    else:
        logger.warning(f"⚠️ Not enough documents to build index for employee {emp['employeeId']}")
        return None

def count_tokens(text: str) -> int:
    """Estimate token count using character-based approximation."""
    return len(text) // 4 + 1
# ────────────────────────────────────────────────────────────────────────────────
#  4.  new query_policy – same signature, no agent changes required
# ────────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
# helper: simple cleaner
_ws_re  = re.compile(r"\s+")
_ctl_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def _clean(text: str) -> str:
    text = _ctl_re.sub(" ", text)         # drop control chars
    return _ws_re.sub(" ", text).strip()  # collapse whitespace


# prompt template – tune wording freely
_SYSTEM_PROMPT = textwrap.dedent("""
You are an intelligent HR assistant AI that provides accurate, professional responses using ONLY the provided context. Follow these rules strictly:
-Answer politely for the greetings messages like hello, hi, how are you, hey ,hola,help Respond with: Hey How can i assist you today, Just dirrectly respond without using context for greetings 

### Context Sections (Use ONLY these to respond according to user query):
1. *POLICY DOCUMENT*: Company policies (ALWAYS prioritize for policy queries) DO Not add Section heading in your response, respond summarized using that section for queries related to company policies
2. *EMPLOYEE-DETAIL*: Employee profiles (use only when explicitly asked)
3. *USER GUIDE DOCUMENT*: EmpowerX User Guide for app usage, signup, login, leave application, employee registration, and project overview, when user ask related to this only add from this section.
4. Data Sections (MUST include ALL records exactly as shown):
   - LEAVE-DETAIL-STATUS → Return as HTML table with ALL rows
   - LEAVE-BALANCE-INFO → Return as HTML list
   - TRAVEL-STATUS-REQUEST → Return as HTML table
   - REIMBURSEMENT-STATUS-INFO → Return as HTML table
   - ATTENDANCE-STATUS-INFO → Return as HTML table
   - TEAM-MEMBER TRAVEL REQUESTS → Return as HTML table

### Critical Rules:
- NEVER skip, summarize, or collapse records - if context has 7 entries, output 7 entries
- NEVER invent information - use ONLY what's in the context
- For leave/travel/reimbursement queries, ALWAYS show ALL matching records
- For team queries, verify manager status before showing reports

### Response Formatting:
1. Policies: Concise paragraph from POLICY-DOCUMENT
   Example: "The casual leave policy states..."
   
2. Employee profiles: Descriptive paragraph from EMPLOYEE-DETAIL
   Example: "John Doe (ID: 123) is a Senior Developer in..."

3. Data requests: ALWAYS use HTML formatting (tables/lists) showing ALL records
   - Leave status: HTML table with ALL leave IDs and details
   - Balances: HTML bullet list
   - Travel/reimbursement: HTML tables

4. If information is missing: "I don't have that information." or "make it conversation by asking "Could you clarify more"

### Final Instructions:
- Be professional and conversational
- Do not include 
- Never add headings like "Answer:" or "User Query:"
- Never reference the context sections in your response
- Do Not include "Context Section", "User query" and any "instructions" provided to you in your final response
Strictly folow :
Do Not add these "instructions" and "what Your task is" in the final response
- If unsure, say "I don't have that information."
""").strip()
_QA_TEMPLATE = PromptTemplate.from_template("""
{system}

Context:
{context}
                                            
*Instructions:* 
-  Never include or add any information or what your task is or add any information that is not related to user query and context.
- if user query and the context is not relatable, Queries like "irrelevant like test, ddc or any irrelevant" just respond politely that you "don't have such information".
- For user guide queries (e.g., how to sign up, login, change password), generate a concise, conversational paragraph without HTML tables or lists.
- Do Not include "Context Section", "User query" and any "instructions" provided to you in your final response
- If unsure, respond: "I don't have that information." or be conversational and ask precisely and accurately to clarify   
- For data tables, use:  
  <table><tr><th>Header1</th>...</tr><tr><td>Data1</td>...</tr>...</table>
- Do not repeat the question. Do not prefix with "User Query", "Answer", or anything else.Just provide the final response withoiut any extra invention, just be conversational
    your final response should be the user faced response with no heading or prefic with "User query", "Answer". just provide the final response                                     
User Query: {question}
Answer:
""".lstrip())

# single LLM chain (reuse global llm)
_hr_chain = LLMChain(llm=llm, prompt=_QA_TEMPLATE)

# ---------------------------------------------------------------------------
def query_policy(state: AgentState) -> AgentState:
 
    logger.debug("query_policy input: %s", state)
    employeeId = state.arguments.get("employeeId")
    email = state.arguments.get("email")
    role = state.role
    org_id = state.arguments.get("org_id")  # if you save this in state in main
    officeId = state.arguments.get("officeId")
    raw_query = state.arguments.get("query", "")
    query = raw_query.strip() if isinstance(raw_query, str) else ""
 
    if not query:
        state.response = "Empty query."
        return state
 
    employee_id = state.arguments.get("employeeId")
    if not employee_id:
        state.response = "No employee ID provided."
        return state
 
    role = state.role or "employee"
    phone_query_pattern = re.compile(
        r"(?:phone\s*number\s*of|contact\s*details\s*of|give\s*me\s*the\s*phone\s*number\s*of|employee\s*contact\s*of)\s*([\w\s]+)",
        re.IGNORECASE
    )
    phone_query_match = phone_query_pattern.search(query)
    is_phone_query = bool(phone_query_match)
    # Check if the query is about reimbursements
    reimbursement_keywords = [
        "reimbursements", "reimbursement details", "expense claims",
        "expense reports", "show my reimbursements", "list my expense claims",
        "view my expense reports"
    ]
    travel_keywords = [
        "travel requests", "travel details", "trips", "show my travel requests",
        "list my trips", "view my travel history"
    ]
    is_reimbursement_query = any(keyword in query.lower() for keyword in reimbursement_keywords)
    is_travel_query = any(keyword in query.lower() for keyword in travel_keywords)
 
    try:
        emp, h = fetch_employee(employee_id, org_id, officeId, role)
        # db     = ensure_index(emp, h)
 
        if is_reimbursement_query:
            # Directly use employee reimbursement data based on employeeId
            reimbursements = emp.get("reimbursements", [])
            if not reimbursements:
                state.response = "No reimbursement records found."
                return state
            response = "<table><tr><th>Reimbursement ID</th><th>Type</th><th>Total</th><th>Status</th></tr>"
            for rb in reimbursements:
                response += (
                    f"<tr><td>{rb['reimbursement_id']}</td>"
                    f"<td>{rb['reimbursement_type']}</td>"
                    f"<td>{rb['total_amount']} {rb['currency']}</td>"
                    f"<td>{rb['status']}</td></tr>"
                )
            response += "</table>"
            state.response = response
            return state
        if is_travel_query:
            # Directly use employee travel request data based on employeeId
            travel_requests = emp.get("travel_requests", [])
            if not travel_requests:
                state.response = "No travel request records found."
                return state
            response = "<table><tr><th>Travel ID</th><th>Type</th><th>City</th><th>Country</th><th>Dates</th><th>Status</th></tr>"
            for tr in travel_requests:
                response += (
                    f"<tr><td>{tr['travel_id']}</td>"
                    f"<td>{tr['travel_type']}</td>"
                    f"<td>{tr['city']}</td>"
                    f"<td>{tr['country']}</td>"
                    f"<td>{tr['from_date']} to {tr['to_date']}</td>"
                    f"<td>{tr['status']}</td></tr>"
                )
            response += "</table>"
            state.response = response
            return state
        # Special handling for leave balance queries
        if "leave balance" in query.lower():
            leave_balances = emp.get("leave_balance", [])
            if not leave_balances:
                state.response = "No leave balance records found."
                return state
            response = "<table><tr><th>Leave Type</th><th>Allocated</th><th>Used</th><th>Remaining</th></tr>"
            for bal in leave_balances:
                response += (
                    f"<tr><td>{bal['leave_type'].capitalize()}</td>"
                    f"<td>{bal['allocated_leaves']}</td>"
                    f"<td>{bal['used_leaves']}</td>"
                    f"<td>{bal['remaining_leaves']}</td></tr>"
                )
            response += "</table>"
            state.response = response
            return state
        if is_phone_query:
            requested_name = phone_query_match.group(1).strip().title()
            if role.lower() not in ["hr", "manager"]:
                state.response = (
                    f"Sorry, only HR or managers can access employee contact details. "
                    f"Please contact HR for assistance with {requested_name}'s contact information."
                )
                return state
 
            # For HR: Check if employee is in the same office
            # For Manager: Check if employee is a direct report
            with conn.cursor() as cur:
                if role.lower() == "hr":
                    cur.execute(
                        """
                        SELECT e.id, e.name, e.phone, e.email, e."officeId"
                        FROM "Employee" e
                        WHERE LOWER(e.name) = LOWER(%s) AND e.is_deleted = FALSE
                        AND e."officeId" = %s
                        """,
                        (requested_name, officeId)
                    )
                elif role.lower() == "manager":
                    cur.execute(
                        """
                        WITH RECURSIVE report_hierarchy AS (
                            SELECT id, name, phone, email, "officeId"
                            FROM "Employee"
                            WHERE id = %s AND is_deleted = FALSE
                            UNION
                            SELECT e.id, e.name, e.phone, e.email, e."officeId"
                            FROM "Employee" e
                            INNER JOIN report_hierarchy rh ON e."managerId" = rh.id
                            WHERE e.is_deleted = FALSE
                        )
                        SELECT id, name, phone, email, "officeId"
                        FROM report_hierarchy
                        WHERE LOWER(name) = LOWER(%s)
                        """,
                        (employee_id, requested_name)
                    )
                result = cur.fetchall()
 
                if not result:
                    state.response = (
                        f"No employee named '{requested_name}' found "
                        f"{'in your office' if role.lower() == 'hr' else 'reporting to you'}."
                    )
                    return state
                elif len(result) > 1:
                    # Multiple matches: Ask for clarification
                    response = "Multiple employees found with similar names. Please specify one of the following:\n"
                    for row in result:
                        response += f"- {row[1]} (ID: {row[0]})\n"
                    state.response = response
                    state.wait_for_input = True
                    state.current_field = "employee_name"
                    state.arguments["employee_name"] = None
                    return state
                else:
                    # Single match: Provide contact details
                    emp_id, name, phone, email, office_id = result[0]
                    state.response = (
                        f"{name}'s contact details:\n"
                        f"Phone: {phone or 'N/A'}\n"
                        f"Email: {email or 'N/A'}"
                    )
                    return state
        # FAISS-based query for other cases (e.g., policy questions)
        org_id = state.arguments.get("org_id")
        officeId = state.arguments.get("officeId")
        db = ensure_index(emp, h , org_id , officeId)
        results = db.similarity_search_with_score(
            query,
            k=CFG.index_k,
            # filter={"employeeId": employee_id}
        )
 
        def apply_filters(results, filters: Dict[str, Any], sim_threshold: float):
            filtered = []
            for doc, score in results:
                try:
                    score = float(score)
                except:
                    continue
                if score < sim_threshold:
                    continue
                if not filters:
                    filtered.append(doc)
                    continue
                match = True
                for k, v in filters.items():
                    doc_val = doc.metadata.get(k)
                    if doc_val is not None and str(doc_val) != str(v):
                        match = False
                        break
                if match:
                    filtered.append(doc)
            return filtered
 
        docs = apply_filters(
            results,
            {"employeeId": employee_id},
            CFG.similarity_threshold
        )
 
        # Fallback to policy documents if no personal data matched
        if not docs:
            raw = cached_similarity_search(query)
            if not raw:
                state.response = "I’m sorry, I don’t have that information."
                return state
            docs = [type("Document", (), {"page_content": c, "metadata": m})() for c, m in raw]
 
        context = "\n\n".join(_clean(d.page_content) for d in docs)[:4096]
        logger.debug("context output: %s", context)
 
        answer = _hr_chain.predict(
            system=_SYSTEM_PROMPT,
            context=context,
            question=query
        ).strip()
        answer = re.sub(r"[ \n]*user\s*query\s*[:：].*", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()
        state.response = answer or "I’m sorry, I don’t have that information."
        logger.debug("query_policy output: %s", state.response)
        return state
 
    except Exception as exc:
        logger.error("query_policy: %s", exc)
        state.response = "Internal HR index error."
        return state