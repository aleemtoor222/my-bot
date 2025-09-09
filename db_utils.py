from datetime import date, datetime
import os
from typing import Any, Dict
import psycopg2, os, json, requests
import logging
from llmconfig import policy_loader, CFG
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Any, Dict, List
from google.oauth2 import service_account
import google.auth.transport.requests
# ---------------------- LOGGING SETUP ----------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_FILE = './firebase_accountkey.json'
SCOPES = ['https://www.googleapis.com/auth/firebase.messaging']
PROJECT_ID = 'hn-employee-hrms-app'

try:
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME", "hrms_empowhr_dev"),
        user=os.getenv("DB_USER", "hnpgadmin"),
        password=os.getenv("DB_PASSWORD", "43ndpzv6TbSHXcZ"),
        host=os.getenv("DB_HOST", "postrgres-hn-dev-uae.postgres.database.azure.com"),
        port=os.getenv("DB_PORT", "5432")
    )
    logger.debug("Database connection successful")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    raise


# try:
#     conn = psycopg2.connect(
#         dbname=os.getenv("DB_NAME", "hrms_empowerhub"),
#         user=os.getenv("DB_USER", "pgadmin"),
#         password=os.getenv("DB_PASSWORD", "8Mh}bsDOav&lAvP72a"),
#         host=os.getenv("DB_HOST", "postgres-hn-stag-uae-01.postgres.database.azure.com"),
#         port=os.getenv("DB_PORT", "5432")
#     )
#     logger.debug("Database connection successful")
# except Exception as e:
#     logger.error(f"Database connection failed: {e}")
#     raise


def build_employee_documents(emp: Dict[str, Any], org_id: int, officeId: int) -> List[Document]:
    """
    Structured, human-readable documents for FAISS, including new sections for reimbursements and travel requests.

    """
    # employeeId = state.arguments.get("employeeId")
    # email = state.arguments.get("email")
    # role = state.role
    # org_id = state.arguments.get("org_id")  # if you save this in state in main
    # officeId = state.arguments.get("officeId")

    eid = emp["employeeId"]
    docs = []
    split = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)

    # Existing sections (unchanged)
    docs.append(Document(
        page_content=(
            "👋 This knowledge base contains personal HR records and policy "
            "documents for efficient Q&A.  Sections are headed by banners like "
            "### LEAVE-DETAIL ### or ### POLICY DOCUMENT ###.  Ask questions "
            "about your leave balance, travel requests, reimbursements, or "
            "company leave policy.or employee contact details."
        ),
        metadata={"section": "intro", "category": "global"}
    ))


    if emp.get("employee_details"):
        docs.append(Document(
            page_content=(
                "### EMPLOYEE DIRECTORY ###\n"
                "This section contains employee profiles including name, ID, department, role,phone number and contact info.\n" 
                "All the information related to employee in Organization. ANy questions like about the phone number of user or where he works at , what its role , who is he"
            ),
            metadata={"section": "Employees_details", "category": "employee_directory","employeeId":eid}
        ))

        for p in emp["employee_details"]:
            doc_text = (
                f"{p['name']} is an employee \n"
                f"They work in the {p['department']} department as a {p['role']} at the {p['office']}]. "
                f"Contact: Email: {p['email']}, Phone Number: {p['phone']}. Gender: {p['gender']}."
            )
            docs.append(Document(
                page_content=doc_text,
                metadata={
                    "section": "EMPLOYEE-DETAIL",
                    "name": p['name'].lower(),
                    "department": p['department'],
                    "role": p['role'],
                    "office": p['office']
                }
            ))


    # docs.append(Document(
    #     page_content=(
    #         "### ORGANIZATION EMPLOYEE DETAIL ###\n"
    #         "This section includes profiles of all employees within the organization. "
    #         "Each block provides the employee's full name, ID, department, role, office, and contact details.For example questions like who is ahsan, who works in hr"
    #     ),
    #     metadata={"section": "intro", "category": "EMPLOYEE-DETAIL","employeeId": eid}
    # ))

    # # One document per employee
    # for employee in emp["employee_details"]:
    #     doc_text = (
    #         f"{p['name']} is an employee with ID {p['id']}. "
    #         f"They work in the {p['department']} department as a {p['role']} at the {p['office']} office. "
    #         f"Their contact details are email: {p['email']}, phone: {p['phone']}. "
    #         f"Their gender is {p['gender']}."
    #     )
    #     docs.append(Document(
    #         page_content=doc_text,
    #         metadata={
    #             "section": "EMPLOYEE-DETAIL",
    #             "name": p['name'].lower(),
    #             "employeeId": p['id'],
    #             "department": p['department'],
    #             "role": p['role'],
    #             "office": p['office']
    #         }
    #     ))
       
    # ATTENDANCE RECORDS
    docs.append(Document(
        page_content=(
            "### ATTENDANCE STATUS INFO ###\n"
            "Contains employee attendance records, including check-in and check-out times."
        ),
        metadata={"section": "intro", "category": "attendance"}
    ))
    try:
        # Fetch attendance data
        url = "http://20.46.54.60:8000/recognise/empowerhub/getuserattendence/"
        headers = {
            "empower_oauth": os.getenv("EMPOWER_OAUTH_TOKEN"),
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Bearer {emp.get('jwt_token')}"  # Assuming token is available
        }
        data = {"email": emp.get("email", "")}
        response = requests.get(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        attendance_data = response.json().get("data", [])
        for record in attendance_data:
            docs.append(Document(
                page_content=(
                    f"### ATTENDANCE RECORD ###\n"
                    f"• Date: {record.get('date', 'N/A')}\n"
                    f"• Check-In: {record.get('check_in_time', 'N/A')}\n"
                    f"• Check-Out: {record.get('check_out_time', 'N/A')}\n"
                    f"• Status: {record.get('status', 'N/A')}"
                ),
                metadata={"section": "attendance", "employeeId": eid, "date": record.get("date")}
            ))
    except Exception as e:
        logger.warning(f"Failed to fetch attendance for indexing: {str(e)}")

    # LEAVE STATUS DETAILS
    docs.append(Document(
        page_content=(
            "###EMPLOYEE LEAVE STATUS DETAILS OVERVIEW ###\n"
            "Use this section for questions like 'what are the status of my leaves' or "
            "'in march my leave status ' or show my leaves taken. or 'Is my leaves approved or not'"
        ),
        metadata={"section": "intro", "category": "leave_details"}
    ))
    for lv in emp["leaves"]:
        docs.append(Document(
            page_content=(
                f"### LEAVE DETAIL STATUS INFO ###\n"
                f"• Leave ID: {lv['leave_id']}\n"
                f"• Type: {lv['leave_type']}\n"
                f"• From: {lv['from_date']}  To: {lv['to_date']}\n"
                f"• Days: {lv['days_count']}   Status: {lv['status']}\n"
                f"• Reason: {lv['reason'] or '—'}"
            ),
           metadata={
            "section": "leave_detail_status",
            "employeeId": eid,
            "leave_id": lv["leave_id"],
            "leave_type": lv["leave_type"],
            "status": lv["status"]
        }
        ))

    # LEAVE BALANCES
    docs.append(Document(
        page_content=(
            "### LEAVE BALANCE INFO ###\n"
            "This section lists allocated, used and remaining leaves per type "
            "and year."
        ),
        metadata={"section": "intro", "category": "leave_balance"}
    ))
    for bal in emp["leave_balance"]:
        docs.append(Document(
            page_content=(
                f"###EMPLOYEE LEAVE-BALANCE INFO ###\n"
                f"• Year: {bal['year']}\n"
                f"• Type: {bal['leave_type']}\n"
                f"• Allocated: {bal['allocated_leaves']}  "
                f"Used: {bal['used_leaves']}  "
                f"Remaining: {bal['remaining_leaves']}"
            ),
            metadata={"section": "leave_balance", "employeeId": eid, "year": bal["year"]}
        ))

    # TRAVEL REQUESTS
    docs.append(Document(
        page_content=(
            "### TRAVEL STATUS REQUESTS ###\n"
            "Contains historical and upcoming travel details."
        ),
        metadata={"section": "intro", "category": "travel_requests"}
    ))
    for tr in emp["travel_requests"]:
        docs.append(Document(
            page_content=(
                f"### EMPLOYEE TRAVEL STATUS REQUEST ###\n"
                f"• Travel ID: {tr['travel_id']}\n"
                f"• Type: {tr['travel_type']}\n"
                f"• City/Country: {tr['city']}, {tr['country']}\n"
                f"• Dates: {tr['from_date']} → {tr['to_date']}\n"
                f"• Purpose: {tr['purpose_of_travel'] or '—'}\n"
                f"• Mode: {tr['mode_of_transport'] or '—'}\n"
                f"• Status: {tr['status']}\n"
                f"• Ref#: {tr['reference_number']}"
            ),
            metadata={"section": "travel_request", "employeeId": eid, "travel_id": tr["travel_id"]}
        ))

    # REIMBURSEMENTS
    docs.append(Document(
        page_content="### REIMBURSEMENT STATUS INFO ###\nPast and pending reimbursements.",
        metadata={"section": "intro", "category": "reimbursements"}
    ))
    for rb in emp["reimbursements"]:
        item_lines = [
            f"  • {it['type']} – {it['scannedAmount']} (img: {it['scannedImage'] or 'none'})"
            for it in rb["itemized_expenses"]
        ] or ["  • No itemised expenses"]
        docs.append(Document(
            page_content=(
                f"### REIMBURSEMENT STATUS INFO ###\n"
                f"• Reimb ID: {rb['reimbursement_id']}\n"
                f"• Type: {rb['reimbursement_type']}\n"
                f"• Period: {rb['start_date']} → {rb['end_date']}\n"
                f"• Total: {rb['total_amount']} {rb['currency']}\n"
                f"• Status: {rb['status']}\n"
                f"• Items:\n" + "\n".join(item_lines)
            ),
            metadata={"section": "reimbursement", "employeeId": eid, "reimbursement_id": rb["reimbursement_id"]}
        ))

    # TEAM LEAVES AND BALANCES (if manager)
    if emp["team_details"]:
        docs.append(Document(
            page_content=(
                "### TEAM-MEMBER TRAVEL AND REIMBURSEMENT REQUESTS ###\n"
                "Visible because you are a manager. Each block describes a direct report’s leave details, including their name, employee ID, and leave records."
            ),
            metadata={"section": "intro", "category": "team_travel_or_reimbursement_requests", "employeeId": eid}
        ))


        # TEAM REIMBURSEMENTS (if manager or HR)
        docs.append(Document(
            page_content=(
                "### TEAM-MEMBER REIMBURSEMENTS REQUESTS ###\n"
                "Visible because you are a manager or HR. Each block describes a direct report’s reimbursement details, including their name, employee ID, and reimbursement records."
            ),
            metadata={"section": "intro", "category": "team_reimbursements", "employeeId": eid}
        ))
        for tm in emp["team_details"]:
            member_id = tm["employee_id"]
            member_data = get_employee_data(member_id, org_id, office_id=officeId, role="employee")
            for rb in member_data.get("reimbursements", []):
                item_lines = [
                    f"  • {it['type']} – {it['scannedAmount']} (img: {it['scannedImage'] or 'none'})"
                    for it in rb["itemized_expenses"]
                ] or ["  • No itemised expenses"]
                docs.append(Document(
                    page_content=(
                        f"### TEAM-MEMBER REIMBURSEMENT ###\n"
                        f"• Employee: {tm['name']} (ID: {tm['employee_id']})\n"
                        f"• Reimb ID: {rb['reimbursement_id']}\n"
                        f"• Type: {rb['reimbursement_type']}\n"
                        f"• Period: {rb['start_date']} → {rb['end_date']}\n"
                        f"• Total: {rb['total_amount']} {rb['currency']}\n"
                        f"• Status: {rb['status']}\n"
                        f"• Items:\n" + "\n".join(item_lines)
                    ),
                    metadata={
                        "section": "team_reimbursement",
                        "employeeId": eid,
                        "team_member_id": tm["employee_id"],
                        "team_member_name": tm["name"],
                        "reimbursement_id": rb["reimbursement_id"]
                    }
                ))

        # TEAM TRAVEL REQUESTS (if manager or HR)
        docs.append(Document(
            page_content=(
                "### TEAM-MEMBER TRAVEL REQUESTS ###\n"
                "Visible because you are a manager or HR. Each block describes a direct report’s travel request details, including their name, employee ID, and travel records."
            ),
            metadata={"section": "intro", "category": "team_travel_requests", "employeeId": eid}
        ))
        for tm in emp["team_details"]:
            member_id = tm["employee_id"]
            member_data = get_employee_data(member_id, org_id, office_id=officeId, role="employee")
            for tr in member_data.get("travel_requests", []):
                docs.append(Document(
                    page_content=(
                        f"### TEAM-MEMBER TRAVEL REQUEST ###\n"
                        f"• Employee: {tm['name']} (ID: {tm['employee_id']})\n"
                        f"• Travel ID: {tr['travel_id']}\n"
                        f"• Type: {tr['travel_type']}\n"
                        f"• City/Country: {tr['city']}, {tr['country']}\n"
                        f"• Dates: {tr['from_date']} → {tr['to_date']}\n"
                        f"• Purpose: {tr['purpose_of_travel'] or '—'}\n"
                        f"• Mode: {tr['mode_of_transport'] or '—'}\n"
                        f"• Status: {tr['status']}\n"
                        f"• Ref#: {tr['reference_number']}"
                    ),
                    metadata={
                        "section": "team_travel_request",
                        "employeeId": eid,
                        "team_member_id": tm["employee_id"],
                        "team_member_name": tm["name"],
                        "travel_id": tr["travel_id"]
                    }
                ))

        # POLICY DOCUMENTS
        docs.append(Document(
            page_content=(
                    "### POLICY AND USER GUIDE DOCUMENTS ###\n"
                    "The following chunks originate from organization-specific HR policy documents or the EmpowerX User Guide. "
                    "Use policy documents for policy-related questions and user guide for app usage, signup, leave application, employee registration, or project overview."
                ),
                metadata={"section": "intro", "category": "policies_and_user_guide", "employeeId": eid}
            ))
        for pol in emp["leave_policies"]:
                try:
                    raw_pages = policy_loader.load_policy(pol["document_url"])[:CFG.policy_limit]
                    for pg in raw_pages:
                        chunks = split.split_text(pg.page_content)
                        for chunk in chunks:
                            section = "user_guide" if pg.metadata.get("policy_type") == "user_guide" else "policy_doc"
                            docs.append(Document(
                                page_content=f"### {'USER GUIDE' if section == 'user_guide' else 'POLICY'} DOCUMENT ###\n" + chunk,
                                metadata={
                                    "section": section,
                                    "policy_title": pg.metadata.get("policy_title", pol["title"]),
                                    "policy_type": pg.metadata.get("policy_type", pol["policy_type"]),
                                    "employeeId": eid
                                }
                            ))
                except Exception as e:
                    logger.warning(f"Policy or user guide {pol['title']} skipped: {e}")

    logger.debug("Employee docs built: %s", len(docs))
    return docs

def get_employee_data(employee_id: int, org_id: int, office_id: int, role: str) -> Dict[str, Any]:
    """
    Retrieve comprehensive employee data, including team details for managers and HR based on office.
    """
    try:
        response = {
            "employeeId": employee_id,
            "leaves": [],
            "leave_balance": [],
            "employee_details":[],
            "travel_requests": [],
            "reimbursements": [],
            "leave_policies": [],
            "team_details": [] if role in ["manager", "hr"] else None
        }


        with conn.cursor() as cur:

            cur.execute(
            """
            SELECT e.id, e.name, e.email, e.phone, e.gender,
                d.name AS department_name,
                r.name AS role_name,
                o.name AS office_name
            FROM "Employee" e
            JOIN "Department" d ON e."departmentId" = d.id
            JOIN "Office" o ON e."officeId" = o.id
            JOIN "Role" r ON e."roleId" = r.id
            WHERE e.is_deleted = FALSE AND e."officeId" IN (
                SELECT id FROM "Office" WHERE "organizationId" = %s
            )
            """,
            (org_id,)
            )

       
            for row in cur.fetchall():
                response["employee_details"].append({    
                        "id": row[0],
                        "name": row[1],
                        "email": row[2] or "N/A",
                        "phone": row[3] or "N/A",
                        "gender": row[4] or "N/A",
                        "department": row[5],
                        "role": row[6],
                        "office": row[7]
                   
                })



            # Check if employee exists
            cur.execute(
                """SELECT id FROM "Employee" WHERE id = %s AND is_deleted = FALSE""",
                (employee_id,)
            )
            if not cur.fetchone():
                raise ValueError(f"No employee found for employeeId: {employee_id}")

            # Determine if employee is a manager or HR
            cur.execute(
                """SELECT r.name
                   FROM "Employee" e
                   JOIN "Role" r ON e."roleId" = r.id
                   WHERE e.id = %s AND e.is_deleted = FALSE""",
                (employee_id,)
            )
            result = cur.fetchone()
            is_manager = bool(result and "manager" in result[0].lower())
            is_hr = bool(result and "hr" in result[0].lower())

            if role.lower() not in ["manager", "hr"]:
                is_manager = False
                is_hr = False
            elif role.lower() == "manager":
                is_manager = True
                is_hr = False
            elif role.lower() == "hr":
                is_manager = False
                is_hr = True

            # Fetch leaves
            cur.execute(
                """SELECT l.id, lt.type AS leave_type, l."fromDate", l."toDate", l.reason, l.days_count, l.status, l.comments
                   FROM "Leave" l
                   JOIN "LeaveType" lt ON l."leaveTypeId" = lt.id
                   WHERE l."employeeId" = %s AND l.is_deleted = FALSE AND lt."organizationId" = %s""",
                (employee_id, org_id)
            )
            for row in cur.fetchall():
                response["leaves"].append({
                    "leave_id": row[0],
                    "leave_type": row[1],
                    "from_date": row[2].isoformat() if row[2] else None,
                    "to_date": row[3].isoformat() if row[3] else None,
                    "reason": row[4],
                    "days_count": row[5],
                    "status": row[6],
                    "comments": row[7]
                })

            # Fetch leave balances
            cur.execute(
                """SELECT COALESCE(lt.type, 'Unknown') AS leave_type,
                          lb.allocated_leaves,
                          lb.remaining_leaves,
                          lb.used_leaves,
                          lb.year
                   FROM "LeavesBalance" lb
                   LEFT JOIN "LeaveType" lt ON lb."leaveTypeId" = lt.id
                       AND lt."organizationId" = %s
                   WHERE lb."employeeId" = %s""",
                (org_id, employee_id)
            )
            for row in cur.fetchall():
                response["leave_balance"].append({
                    "leave_type": row[0],
                    "allocated_leaves": row[1],
                    "remaining_leaves": row[2],
                    "used_leaves": row[3],
                    "year": row[4]
                })

            # Fetch travel requests
            cur.execute(
                """SELECT tr.id, tt.type AS travel_type, tr.city, tr.country, tr."fromDate", tr."toDate",
                          tr."purposeOfTravel", tr."modeOfTransport", tr.status, tr.comments, tr.reference_number
                   FROM "TravelRequest" tr
                   JOIN "TravelType" tt ON tr."travelTypeId" = tt.id
                   WHERE tr."employeeId" = %s AND tr.is_deleted = FALSE AND tt."organizationId" = %s""",
                (employee_id, org_id)
            )
            for row in cur.fetchall():
                response["travel_requests"].append({
                    "travel_id": row[0],
                    "travel_type": row[1],
                    "city": row[2],
                    "country": row[3],
                    "from_date": row[4].isoformat() if row[4] else None,
                    "to_date": row[5].isoformat() if row[5] else None,
                    "purpose_of_travel": row[6],
                    "mode_of_transport": row[7],
                    "status": row[8],
                    "comments": row[9],
                    "reference_number": row[10]
                })

            # Fetch reimbursements
            cur.execute(
                """SELECT r.id, rt.type AS reimbursement_type, r."purposeofExpense", r."startDate", r."endDate",
                          r."itemizedExpenses", r.totalamount, r.currency, r.image, r.status
                   FROM "Reimbursement" r
                   JOIN "ReimbursementType" rt ON r."reimbursementTypeId" = rt.id
                   WHERE r."employeeId" = %s AND r.is_deleted = FALSE AND rt."organizationId" = %s""",
                (employee_id, org_id)
            )
            for row in cur.fetchall():
                response["reimbursements"].append({
                    "reimbursement_id": row[0],
                    "reimbursement_type": row[1],
                    "purpose_of_expense": row[2],
                    "start_date": row[3].isoformat() if row[3] else None,
                    "end_date": row[4].isoformat() if row[4] else None,
                    "itemized_expenses": row[5] if isinstance(row[5], list) else json.loads(row[5]) if row[5] else [],
                    "total_amount": float(row[6]) if row[6] else None,
                    "currency": row[7],
                    "image": row[8],
                    "status": row[9]
                })

            # Fetch leave policies
            cur.execute(
                """SELECT title, document_url, COALESCE(policy_type, 'unknown') AS policy_type
                   FROM "PolicyDocument"
                   WHERE "organizationId" = %s AND is_deleted = FALSE""",
                (org_id,)
            )
            for row in cur.fetchall():
                response["leave_policies"].append({
                    "title": row[0],
                    "document_url": row[1],
                    "policy_type": row[2]
                })

            # Fetch team details if manager or HR
        if is_manager or is_hr:
            team_details = []

            def fetch_team(emp_id: int, depth: int = 0, max_depth: int = 10) -> list:
                if depth > max_depth:
                    logger.warning(f"Max recursion depth reached for employee_id={emp_id}")
                    return []

                try:
                    with conn.cursor() as sub_cur:  # Create new cursor for each recursive call
                        sub_cur.execute(
                            """
                            SELECT e.id, e.name, e.email, r.name AS role_name,
                                COALESCE(e."managerId", 0) AS manager_id,
                                (SELECT name FROM "Employee" WHERE id = e."managerId" AND is_deleted = FALSE) AS manager_name
                            FROM "Employee" e
                            JOIN "Role" r ON e."roleId" = r.id
                            WHERE e."managerId" = %s
                            AND e.is_deleted = FALSE
                            AND e."officeId" IN (
                                SELECT id FROM "Office" WHERE "organizationId" = %s
                            )
                            """,
                            (emp_id, org_id)
                        )
                        direct_reports = sub_cur.fetchall()
                        result = []

                        for report in direct_reports:
                            member_id, name, email, role_name, manager_id, manager_name = report
                            sub_cur.execute(  # Use sub_cur for leaves
                                """
                                SELECT l.id, lt.type AS leave_type, l."fromDate", l."toDate", l.reason, l.days_count, l.status
                                FROM "Leave" l
                                JOIN "LeaveType" lt ON l."leaveTypeId" = lt.id
                                WHERE l."employeeId" = %s AND l.is_deleted = FALSE AND lt."organizationId" = %s
                                """,
                                (member_id, org_id)
                            )
                            member_leaves = [
                                {
                                    "leave_id": row[0],
                                    "leave_type": row[1],
                                    "from_date": row[2].isoformat() if row[2] else None,
                                    "to_date": row[3].isoformat() if row[3] else None,
                                    "reason": row[4],
                                    "days_count": row[5],
                                    "status": row[6]
                                } for row in sub_cur.fetchall()
                            ]
                            sub_cur.execute(  # Use sub_cur for leave balances
                                """
                                SELECT COALESCE(lt.type, 'Unknown') AS leave_type,
                                    lb.allocated_leaves,
                                    lb.remaining_leaves,
                                    lb.used_leaves,
                                    lb.year
                                FROM "LeavesBalance" lb
                                LEFT JOIN "LeaveType" lt ON lb."leaveTypeId" = lt.id
                                    AND lt."organizationId" = %s
                                WHERE lb."employeeId" = %s
                                """,
                                (org_id, member_id)
                            )
                            member_leave_balances = [
                                {
                                    "leave_type": row[0],
                                    "allocated_leaves": row[1],
                                    "remaining_leaves": row[2],
                                    "used_leaves": row[3],
                                    "year": row[4]
                                } for row in sub_cur.fetchall()
                            ]
                            member = {
                                "employee_id": member_id,
                                "name": name,
                                "email": email,
                                "role": role_name,
                                "reports_to_id": manager_id,
                                "reports_to_name": manager_name or "None",
                                "depth": depth + 1,
                                "leaves": member_leaves,
                                "leave_balance": member_leave_balances
                            }
                            result.append(member)

                            # Recursively fetch team if the report is a manager
                            if "manager" in role_name.lower():
                                sub_team = fetch_team(member_id, depth + 1, max_depth)
                                result.extend(sub_team)

                        return result

                except Exception as e:
                    logger.error(f"fetch_team error for employee_id={emp_id}: {str(e)}")
                    return []

            if is_manager:
                team_details = fetch_team(employee_id)
            elif is_hr:
                
                with conn.cursor() as cur:


                    cur.execute(
                        """
                        SELECT e.id, e.name, e.email
                        FROM "Employee" e
                        WHERE e."officeId" = %s AND e.id != %s AND e.is_deleted = FALSE
                        AND e."officeId" IN (
                            SELECT id FROM "Office" WHERE "organizationId" = %s
                        )
                        """,
                        (office_id, employee_id, org_id)
                    )
                    team_members = cur.fetchall()
                    for member in team_members:
                        member_id, name, email = member
                        cur.execute(
                            """
                            SELECT l.id, lt.type AS leave_type, l."fromDate", l."toDate", l.reason, l.days_count, l.status
                            FROM "Leave" l
                            JOIN "LeaveType" lt ON l."leaveTypeId" = lt.id
                            WHERE l."employeeId" = %s AND l.is_deleted = FALSE AND lt."organizationId" = %s
                            """,
                            (member_id, org_id)
                        )
                        member_leaves = [
                            {
                                "leave_id": row[0],
                                "leave_type": row[1],
                                "from_date": row[2].isoformat() if row[2] else None,
                                "to_date": row[3].isoformat() if row[3] else None,
                                "reason": row[4],
                                "days_count": row[5],
                                "status": row[6]
                            } for row in cur.fetchall()
                        ]
                        cur.execute(
                            """
                            SELECT COALESCE(lt.type, 'Unknown') AS leave_type,
                                lb.allocated_leaves,
                                lb.remaining_leaves,
                                lb.used_leaves,
                                lb.year
                            FROM "LeavesBalance" lb
                            LEFT JOIN "LeaveType" lt ON lb."leaveTypeId" = lt.id
                                AND lt."organizationId" = %s
                            WHERE lb."employeeId" = %s
                            """,
                            (org_id, member_id)
                        )
                        member_leave_balances = [
                            {
                                "leave_type": row[0],
                                "allocated_leaves": row[1],
                                "remaining_leaves": row[2],
                                "used_leaves": row[3],
                                "year": row[4]
                            } for row in cur.fetchall()
                        ]
                        team_details.append({
                            "employee_id": member_id,
                            "name": name,
                            "email": email,
                            "role": "Unknown",
                            "reports_to_id": 0,
                            "reports_to_name": "None",
                            "depth": 1,
                            "leaves": member_leaves,
                            "leave_balance": member_leave_balances
                        })

            response["team_details"] = team_details

        return response

    except Exception as e:
        logger.error(f"Failed to retrieve employee data for employeeId {employee_id}: {str(e)}")
        raise

def is_clock_in_allowed(cur, org_id: int) -> Dict[str, Any]:
    cur.execute("""
        SELECT f.id
        FROM "Feature" f
        JOIN "OrganizationFeature" of ON f.id = of."featureId"
        WHERE f.key = 'clock_in'
        AND of."organizationId" = %s
        AND f."isDeleted" = false AND of."isDeleted" = false
        AND of."isEnabled" = true
    """, (org_id,))
    row = cur.fetchone()
    if not row:
        return False, False
    feature_id = row[0]
 
    cur.execute("""
        SELECT 1
        FROM "SubFeature" sf
        JOIN "OrganizationSubFeature" osf ON sf.id = osf."subFeatureId"
        WHERE sf.key = 'geolocation'
        AND sf."featureId" = %s
        AND osf."organizationId" = %s
        AND sf."isDeleted" = false AND osf."isDeleted" = false
        AND osf."isEnabled" = true
    """, (feature_id, org_id))
    geo_enabled = cur.fetchone() is not None
    return True, geo_enabled



def get_fcm_access_token():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    return credentials.token

def send_push_notification(fcm_token, title, body):
    access_token = get_fcm_access_token()
    logger.info("accessToken:{access_token}")
    url = f'https://fcm.googleapis.com/v1/projects/hn-employee-hrms-app/messages:send'

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    payload = {
        "message": {
            "token": fcm_token,
            "notification": {
                "title": title,
                "body": body
            },
            "data": {
                "title": title,
                "body": body
            }
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("Notification sent successfully")
        return True
    
    else:
        print(f"Failed to send notification: {response.text}")
        return False
    
def get_employee_names_for_org(conn, org_id):
    logger.debug(f"[NameExtractor] Fetching employees for org_id={org_id}")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT e.name
            FROM "Employee" e
            INNER JOIN "Office" o ON e."officeId" = o.id
            WHERE o."organizationId" = %s
            AND e.is_deleted = false
        """, (org_id,))
        results = [row[0] for row in cur.fetchall()]
        logger.debug(f"[NameExtractor] Found employees: {results}")
        return results




