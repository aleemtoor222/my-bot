from llmconfig import AgentState ,generate_pdf_and_upload_to_blob
from datetime import date, datetime, timezone, timedelta
from dateutil import parser as dateutil_parser
import time
import logging
import requests, json, re, textwrap
from difflib import SequenceMatcher
from flask import request
import os
import pytz
from collections import defaultdict
from db_utils import get_employee_data, conn , send_push_notification, is_clock_in_allowed
from typing import List, Dict
from pdf_generator import build_team_timesheet_pdf
# from agent_graph import send_push_notification
# ---------------------- LOGGING SETUP ----------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

timeout_seconds = 60

def handle_user_input(state: AgentState, user_input: str) -> AgentState:
    now = datetime.utcnow()
    if (now - state.timestamp).total_seconds() > timeout_seconds:
        state.response = "⏳ Timeout. Please start over."
        return AgentState(user_input="")
        # User entered something unexpected — guide them instead
    state.response = (
        "🤖 I didn't quite catch that. But don't worry!\n"
        "Just type what you'd like to do — for example:\n"
        "• Apply for leave\n"
        "• Submit a reimbursement\n"
        "• Request a business trip\n"
        "• Ask about a policy or HR guideline\n\n"
        "I'm here to help!"
    )
        
    return reset_state(state)

def handle_cancel_input(state, cancel_message="❌ Action cancelled."):
    user_input = state.user_input.strip().lower()
    if user_input == "cancel":
        state.response = cancel_message
        return reset_state(state)
    return None  # Not cancelled, continue normal flow


def format_duration(dur):
    if not dur:
        return "—"
    hours = dur // 60
    minutes = dur % 60
    if hours > 0:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    return f"{minutes}m"



def respond(state: AgentState) -> AgentState:
    logger.debug(f"respond input: {state}")
    # Initialize response if empty
    if not state.response and not state.ask_user:
        state.response = "Hey there! how can I assist you today?"

    # Check if the function completed successfully and needs a state reset
    if (
        state.function_name in ["apply_leave", "apply_travel", "apply_reimbursement"]
        and not state.validated.get("pending_confirmation", False)
        and state.response.startswith(("Leave recorded", "Travel request created", "✅ Reimbursement logged"))
    ):
        logger.debug("respond: Resetting state after successful completion, preserving success message.")
        # Store the success message before resetting
        success_message = state.response
        # Reset state but preserve email, employeeId, and role
        new_state = AgentState(
            user_input="",
            role=state.role,
            arguments={"email": state.arguments.get("email"), "employeeId": state.arguments.get("employeeId")},
            response=success_message,  # Carry forward the success message
            ask_user=""  # Add contextual follow-up
        )
        logger.debug(f"respond output: {new_state}")
        return new_state
    informational_fns = {
        "query_policy",
        "show_team",
        "team_leaves",
        "get_attendance",
        "get_manager_detail",
        "get_team_attendance",
        "check_in_to_project"
    }

    if state.function_name in informational_fns and state.response:
        logger.debug(f"respond: Resetting state after {state.function_name} completion, preserving answer.")
        info_answer = state.response
        new_state = AgentState(
            user_input="",
            role=state.role,
            arguments={
                "email": state.arguments.get("email"),
                "employeeId": state.arguments.get("employeeId")
            },
            response=info_answer,
            ask_user=""
        )
        logger.debug(f"respond output (after info): {new_state}")
        return new_state


    logger.debug(f"respond output: {state}")
    return state

def reset_state(state: AgentState) -> AgentState:
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


def show_team(state: AgentState) -> AgentState:
    """
    Fetch and display team members for a manager or HR in HTML table format, based on employee name.
    """
    logger.debug(f"show_team input: {state}")
    if state.role not in ["manager", "hr"]:
        state.response = "Only managers or HR can view team details."
        return state
 
    try:
        employee_name = state.arguments.get("employee_name")
        # If employee_name is a list like ['Ali Raja'], convert to string
        if isinstance(employee_name, (list, tuple)):
            # Remove any None or empty strings and join with space
            employee_name = " ".join(filter(None, map(str, employee_name))).strip()

        employee_id = state.arguments.get("employeeId")
        org_id = state.arguments.get("org_id")  # if you save this in state in main
        office_id = state.arguments.get("officeId")
        print("org_id",org_id)
        # If no employee_name is provided, use the logged-in user's employeeId
        if not employee_name:
            employee_id = state.arguments.get("employeeId")
            if employee_id is None:
                state.response = "No employee name provided and no logged-in user context available."
                logger.error("show_team: No employee name or ID provided")
                return state
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e.id, e.name
                    FROM "Employee" e
                    JOIN "Office" o ON e."officeId" = o.id
                    WHERE e.id = %s
                    AND e.is_deleted = FALSE
                    AND o."organizationId" = %s

                    """,
                    (employee_id, org_id)
                )
                match = cur.fetchone()
                if not match:
                    state.response = f"No employee found with ID {employee_id}."
                    logger.debug(f"show_team: No employee found for ID {employee_id}")
                    return state
                employee_id, employee_name = match
                logger.debug(f"show_team: Using logged-in employee '{employee_name}' with ID {employee_id}")
        else:
            # Search for employee by name
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e.id, e.name
                    FROM "Employee" e
                    JOIN "Office" o ON e."officeId" = o.id
                    WHERE e.name ILIKE %s
                    AND e.is_deleted = FALSE
                    AND o."organizationId" = %s

                    """,
                    (f'%{employee_name}%', org_id)
                )
                matches = cur.fetchall()
               
                if not matches:
                    state.response = f"No employee found with name '{employee_name}'."
                    logger.debug(f"show_team: No employee found for name '{employee_name}'")
                    return state
                elif len(matches) > 1:
                    # Handle multiple matches by prompting for clarification
                    response = "Multiple employees found with similar names. Please select one by providing the full name:<br><ul>"
                    for emp_id, full_name in matches:
                        response += f"<li>{full_name} (ID: {emp_id})</li>"
                    response += "</ul>"
                    state.response = response
                    state.wait_for_input = True
                    logger.debug(f"show_team: Multiple employees found for name '{employee_name}': {[m[1] for m in matches]}")
                    return state
                else:
                    # Single match found
                    employee_id, employee_name = matches[0]
                    logger.debug(f"show_team: Matched manager '{employee_name}' to ID {employee_id} for fetching team members")
 
        # Log the employee_id to confirm it's correct
        logger.debug(f"show_team: Fetching team members for employee_id {employee_id}")
 
        # Fetch team members where managerId matches the employee's ID
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, email
                FROM "Employee"
                WHERE "managerId" = %s
                AND is_deleted = FALSE
                """,
                (employee_id,)
            )
            team = cur.fetchall()
 
        if not team:
            state.response = f"No team members assigned to {employee_name} (ID: {employee_id})."
            logger.debug(f"show_team: No team members for employee ID {employee_id}")
            return state
 
        # Log the number of team members found
        logger.debug(f"show_team: Found {len(team)} team members for {employee_name} (ID: {employee_id})")
 
        # Build HTML table for team members
        response = """
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Email</th>
                    </tr>
                </thead>
                <tbody>
        """
        for member_id, member_name, member_email in team:
            response += textwrap.dedent(f"""\
                <tr>
                    <td>{member_name}</td>
                    <td>{member_email}</td>
                </tr>
            """).strip()
        response += "</tbody></table>"
        state.response = response
        state.wait_for_input = False
        logger.debug(f"show_team output: {state}")
        return state
 
    except Exception as e:
        logger.error(f"show_team error: {str(e)}")
        state.response = f"Failed to fetch team details: {str(e)}"
        return state


def team_leaves(state: AgentState) -> AgentState:
    employeeId = state.arguments.get("employeeId")
    role = state.role
    org_id = state.arguments.get("org_id")
    officeId = state.arguments.get("officeId")

    logger.debug(f"team_leaves input: {state}")
    if role not in ["manager", "hr"]:
        state.response = "Only managers or HR can view team member leave details."
        return state

    employee_name = state.arguments.get("employee_name")
    leave_status = state.arguments.get("status")
    start_date = state.arguments.get("start_date")
    end_date = state.arguments.get("end_date")

    try:
        emp_data = get_employee_data(employeeId, org_id, office_id=officeId, role=role)
        team = emp_data.get("team_details", [])
        if not team:
            state.response = "You have no team members assigned."
            return state

        members_to_show = []
        if employee_name:
            for m in team:
                if m["name"].lower() == employee_name.lower():
                    members_to_show = [m]
                    break
                similarity = SequenceMatcher(None, m["name"].lower(), employee_name.lower()).ratio()
                if similarity > 0.9:
                    members_to_show = [m]
                    break
            if not members_to_show:
                team_names = [m["name"] for m in team]
                state.response = (
                    f"'{employee_name}' is not a member of your team.<br>"
                    f"Your team members are: {', '.join(team_names)}."
                )
                state.ask_user = "Please provide a valid team member’s name:"
                state.wait_for_input = True
                state.current_field = "employee_name"
                return state
        else:
            members_to_show = team

        response = """
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Email</th>
                        <th>Leave Details</th>
                        <th>Leave Balances</th>
                    </tr>
                </thead>
                <tbody>
        """

        for member in members_to_show:
            leave_details = ""
            leaves = member.get("leaves", [])
            if leave_status:
                leaves = [l for l in leaves if l["status"].lower() == leave_status.lower()]
            if start_date:
                leaves = [l for l in leaves if l["from_date"] >= str(start_date)]
            if end_date:
                leaves = [l for l in leaves if l["to_date"] <= str(end_date)]

            if not leaves:
                leave_details = "No matching leaves found."
            else:
                for leave in leaves:
                    leave_details += textwrap.dedent(f"""\
                        Leave ID: {leave['leave_id']}, Type: {leave['leave_type']},
                        From: {leave['from_date']} to {leave['to_date']},
                        Days: {leave['days_count']}, Status: {leave['status']}<br>
                    """).strip() + "<br>"

            leave_balances = ""
            balances = member.get("leave_balance", [])
            if not balances:
                leave_balances = "No leave balances recorded."
            else:
                for bal in balances:
                    leave_balances += textwrap.dedent(f"""\
                        {bal['leave_type']} ({bal['year']}):
                        {bal['remaining_leaves']}/{bal['allocated_leaves']} remaining
                        (used {bal['used_leaves']})<br>
                    """).strip() + "<br>"

            response += textwrap.dedent(f"""\
                <tr>
                    <td>{member['name']}</td>
                    <td>{member['email']}</td>
                    <td>{leave_details}</td>
                    <td>{leave_balances}</td>
                </tr>
            """).strip()

        response += "</tbody></table>"
        state.response = response
        state.wait_for_input = False
        state.current_field = None
        logger.debug(f"team_leaves output: {state}")
        return state

    except Exception as e:
        logger.error(f"team_leaves error: {str(e)}")
        state.response = f"Failed to fetch leave details: {str(e)}"
        return state

def get_attendance(state: AgentState) -> AgentState:
    """
    Fetch and display attendance details for the employee based on their email from the state.
    """

    employeeId = state.arguments.get("employeeId")
    email = state.arguments.get("email")
    role = state.role
    org_id = state.arguments.get("org_id")  # if you save this in state in main
    officeId = state.arguments.get("officeId")

    logger.debug(f"get_attendance input: {state}")
    try:
        # Get email from state
        email = state.arguments.get("email")
        if not email:
            state.response = "No email found in state."
            return state

        # Get Authorization header for API call
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            state.response = "Missing or invalid Authorization header."
            return state
        token = auth_header.split(" ")[1]

        # Prepare API request
        url = "http://20.46.54.60:8000/recognise/empowerhub/getuserattendence/"
        headers = {
            "empower_oauth": os.getenv("EMPOWER_OAUTH_TOKEN"),
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Bearer {token}"
        }
        data = {"email": email}

        # Make the API call
        response = requests.get(url, headers=headers, data=data, timeout=10)
        logger.debug(f"API response status: {response.status_code}")
        logger.debug(f"API response content: {response.text}")
        response.raise_for_status()
        attendance_data = response.json()
        logger.debug(f"Parsed attendance data: {attendance_data}")

        # Check if response is empty or lacks expected structure
        if not attendance_data or "attendance_records" not in attendance_data:
            state.response = "No attendance records found."
            return state

        records = attendance_data.get("attendance_records", [])
        if not isinstance(records, list) or not records:
            state.response = "No attendance records found (empty or invalid data)."
            return state

        # Format response as HTML table
        response_html = """
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Check-In Time</th>
                        <th>Check-Out Time</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        for record in records:
            # Map API fields to expected fields
            check_in_time = record.get("timein", "N/A")
            check_out_time = record.get("timeout", "N/A")
            status = "Present" if check_in_time != "N/A" and check_out_time == "N/A" else "Completed" if check_out_time != "N/A" else "N/A"
           
            response_html += textwrap.dedent(f"""\
                <tr>
                    <td>{record.get('date', 'N/A')}</td>
                    <td>{check_in_time}</td>
                    <td>{check_out_time}</td>
                    <td>{status}</td>
                </tr>
            """).strip()
        response_html += "</tbody></table>"
        state.response = response_html
        state.wait_for_input = False
        logger.debug(f"get_attendance output: {state}")
        return state

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch attendance: {str(e)}")
        state.response = f"Failed to fetch attendance data: {str(e)}"
        return state
    except Exception as e:
        logger.error(f"get_attendance error: {str(e)}")
        state.response = f"An error occurred: {str(e)}"
        return state

def get_team_attendance(state: AgentState) -> AgentState:
    """
    Fetch and display team attendance details for a manager based on date range and optional person name.
    """
    employeeId = state.arguments.get("employeeId")
    email = state.arguments.get("email")
    role = state.role
    org_id = state.arguments.get("org_id")  # if you save this in state in main
    officeId = state.arguments.get("officeId")

    logger.debug(f"get_team_attendance input: {state}")
    if state.role.lower() != "manager":
        state.response = "Only managers can view team attendance details."
        state.wait_for_input = False
        return state
    try:
        # Get current date for default if no dates provided
        current_date = datetime.now().date()
        person_name = state.arguments.get("personName")
        emails = []
 
        with conn.cursor() as cur:
            if person_name:
                # Search for employees by partial name (case-insensitive)
                cur.execute(
                    """SELECT email, name FROM "Employee"
                       WHERE LOWER(name) LIKE LOWER(%s) AND "managerId" = %s
                       AND is_deleted = FALSE AND "officeId" = %s""",
                    (f"%{person_name}%", state.arguments.get("employeeId", employeeId), officeId)
                )
                results = cur.fetchall()
                if not results:
                    state.response = f"No employee found with name containing '{person_name}'."
                    state.wait_for_input = False
                    return state
                elif len(results) == 1:
                    emails = [results[0][0]]  # Single match: use email
                else:
                    # Multiple matches: prepare suggestion prompt
                    suggestions = [row[1] for row in results]
                    suggestion_text = "Multiple employees found with similar names. Please specify one of the following:\n"
                    for name in suggestions:
                        suggestion_text += f"- {name}\n"
                    state.response = suggestion_text
                    state.wait_for_input = True
                    state.current_field = "personName"
                    state.already_asked.add("personName")
                    logger.debug(f"Multiple employees found: {suggestions}")
                    return state
            else:
                # Fetch all team emails
                cur.execute(
                    """SELECT email FROM "Employee"
                       WHERE "managerId" = %s AND is_deleted = FALSE AND "officeId" = %s""",
                    (state.arguments.get("employeeId", employeeId), officeId)
                )
                emails = [row[0] for row in cur.fetchall()]
 
        logger.info(f"teamemails: {emails}")
        if not emails:
            state.response = "You have no team members assigned."
            state.wait_for_input = False
            return state
 
        # Get Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            state.response = "Missing or invalid Authorization header."
            state.wait_for_input = False
            return state
        token = auth_header.split(" ")[1]
 
        # Validate and convert date strings to date objects
        start_date_str = state.arguments.get("start_date")
        end_date_str = state.arguments.get("end_date")
        try:
            if start_date_str and end_date_str:
                start_date = dateutil_parser.parse(start_date_str).date()
                end_date = dateutil_parser.parse(end_date_str).date()
                if end_date < start_date:
                    state.response = "End date must be on or after start date."
                    state.wait_for_input = False
                    return state
            else:
                # Use current date if no dates provided
                start_date = current_date
                end_date = current_date
        except (ValueError, TypeError):
            state.response = "Invalid date format. Please use YYYY-MM-DD."
            state.wait_for_input = False
            return state
 
        # API request
        url = "http://20.46.54.60:8000/recognise/empowerhub/getteamattendence/"
        headers = {
            "empower_oauth": os.getenv("EMPOWER_OAUTH_TOKEN"),
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "emails": json.dumps(emails),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
 
        # Make API call
        response = requests.get(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        attendance_data = response.json()
 
        # Extract records from nested team_attendance
        records = []
        for team_member in attendance_data.get("team_attendance", []):
            records.extend(team_member.get("attendance_records", []))
 
        if not records:
            state.response = "No attendance records found for the specified employee(s) in the date range."
            state.wait_for_input = False
            return state
 
        # Format as HTML table
        response_html = """
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Date</th>
                        <th>Check-In Time</th>
                        <th>Check-Out Time</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        for record in records:
            status = "Present" if record.get("timein") and not record.get("timeout") else "Completed" if record.get("timeout") else "N/A"
            response_html += textwrap.dedent(f"""\
                <tr>
                    <td>{record.get('person_name', 'N/A')}</td>
                    <td>{record.get('date', 'N/A')}</td>
                    <td>{record.get('timein', 'N/A')}</td>
                    <td>{record.get('timeout', 'N/A')}</td>
                    <td>{status}</td>
                </tr>
            """).strip()
        response_html += "</tbody></table>"
        state.response = response_html
        state.wait_for_input = False
        logger.debug(f"get_team_attendance output: {state}")
        return state
 
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch team attendance: {str(e)}")
        state.response = f"Failed to fetch team attendance data: {str(e)}"
        state.wait_for_input = False
        return state
    except Exception as e:
        logger.error(f"get_team_attendance error: {str(e)}")
        state.response = f"An error occurred: {str(e)}"
        state.wait_for_input = False
        return state
def get_manager_detail(state: AgentState) -> AgentState:
    """
    Fetch and display manager details for the requesting employee or a specified employee in description form.
    For 'show my manager', use the requesting employee's managerId.
    For 'show [employee_name]'s manager', verify HR role and office, then fetch manager details.
    """
    logger.debug(f"get_manager_detail input: {state}")
    employee_id = state.arguments.get("employeeId")
    employee_name = state.arguments.get("employee_name")
    logger.debug(f"employee_id: {employee_id}, employee_name: {employee_name}")
    role = state.role.lower()
    # Safely access officeId if defined in AgentState, else None
    office_id = getattr(state, "officeId", None)

    if not employee_id:
        state.response = "Employee ID is missing."
        state.wait_for_input = False
        return state

    try:
        with conn.cursor() as cur:
            # Case 1: "show my manager" (no employee_name or "my")
            if not employee_name or employee_name.lower() == "my":
                cur.execute(
                    """
                    SELECT m.id, m.name, m.email, m.phone, d.name AS department_name, r.name AS role_name, o.name AS office_name
                    FROM "Employee" e
                    LEFT JOIN "Employee" m ON e."managerId" = m.id
                    JOIN "Department" d ON m."departmentId" = d.id
                    JOIN "Role" r ON m."roleId" = r.id
                    JOIN "Office" o ON m."officeId" = o.id
                    WHERE e.id = %s AND e.is_deleted = FALSE AND (m.is_deleted = FALSE OR m.id IS NULL)
                    """,
                    (employee_id,)
                )
                manager = cur.fetchone()
                if not manager:
                    state.response = "No manager assigned to you."
                    state.wait_for_input = False
                    return state

                # Format manager details as a description
                response = (
                    f"Your manager is {manager[1]}\n"
                    f"Email: {manager[2] or 'N/A'}\n"
                    f"Phone: {manager[3] or 'N/A'}\n"
                    f"Department: {manager[4]}\n"
                    f"Office: {manager[6]}"
                )
                state.response = response
                state.wait_for_input = False
                logger.debug(f"get_manager_detail output: {state}")
                return state

            # Case 2: "show [employee_name]'s manager" (HR only)
            if role != "hr":
                state.response = "Only HR can view another employee's manager details."
                state.wait_for_input = False
                return state

            if not office_id:
                state.response = "Office ID is missing for HR query."
                state.wait_for_input = False
                return state

            cur.execute(
                """
                SELECT e.id, e.name, e."officeId", m.id, m.name, m.email, m.phone, d.name AS department_name, r.name AS role_name, o.name AS office_name
                FROM "Employee" e
                LEFT JOIN "Employee" m ON e."managerId" = m.id
                JOIN "Department" d ON m."departmentId" = d.id
                JOIN "Role" r ON m."roleId" = r.id
                JOIN "Office" o ON m."officeId" = o.id
                WHERE LOWER(e.name) = LOWER(%s) AND e.is_deleted = FALSE AND (m.is_deleted = FALSE OR m.id IS NULL)
                """,
                (employee_name,)
            )
            result = cur.fetchone()
            if not result:
                state.response = f"No employee named '{employee_name}' found."
                state.wait_for_input = False
                return state

            emp_id, emp_name, emp_office_id, manager_id, manager_name, manager_email, manager_phone, dept_name, role_name, office_name = result

            if emp_office_id != office_id:
                state.response = f"'{employee_name}' is not in your office."
                state.wait_for_input = False
                return state

            if not manager_id:
                state.response = f"No manager assigned to {employee_name}."
                state.wait_for_input = False
                return state

            # Format manager details as a description
            response = (
                f"The manager of {employee_name} is {manager_name} (ID: {manager_id}). "
                f"Email: {manager_email or 'N/A'}, Phone: {manager_phone or 'N/A'}. "
                f"Department: {dept_name}, Role: {role_name}, Office: {office_name}."
            )
            state.response = response
            state.wait_for_input = False
            logger.debug(f"get_manager_detail output: {state}")
            return state

    except Exception as e:
        logger.error(f"get_manager_detail error: {str(e)}")
        state.response = f"Failed to fetch manager details: {str(e)}"
        state.wait_for_input = False
        return state
    

#  Clockin/out feature ----

#--------------------------------------------------------------- Helper functions ---------------------------- 


def get_project_id_by_name(cur, emp_id, name):
    cur.execute("""
        SELECT p.id, p.name FROM "Project" p
        JOIN "EmployeeProject" ep ON ep."projectId" = p.id
        WHERE ep."employeeId" = %s AND p."isDeleted" = FALSE AND ep."isDeleted" = FALSE
        AND p."name" ILIKE %s LIMIT 1
    """, (emp_id, f"%{name}%"))
    row = cur.fetchone()
    return (row[0], row[1]) if row else (None, None)


def get_assigned_projects(cur, emp_id):
    cur.execute("""
        SELECT p.name FROM "Project" p
        JOIN "EmployeeProject" ep ON ep."projectId" = p.id
        WHERE ep."employeeId" = %s AND p."isDeleted" = FALSE AND ep."isDeleted" = FALSE
    """, (emp_id,))
    return [r[0] for r in cur.fetchall()]


def is_already_checked_in(cur, emp_id):
    cur.execute("""
        SELECT id FROM "ClockIn"
        WHERE "employeeId" = %s AND DATE("startTime") = CURRENT_DATE
        AND "endTime" IS NULL AND "isDeleted" = FALSE
    """, (emp_id,))
    return cur.fetchone() is not None


def get_active_project_name(cur, emp_id):
    cur.execute("""
        SELECT "projectId", "startTime" FROM "ClockIn"
        WHERE "employeeId" = %s AND DATE("startTime") = CURRENT_DATE
        AND "endTime" IS NULL AND "isDeleted" = FALSE
    """, (emp_id,))
    active = cur.fetchone()
    if not active:
        return {"name": "Unknown", "start": ""}
    cur.execute('SELECT name FROM "Project" WHERE id = %s', (active[0],))
    name = cur.fetchone()[0]
    return {"name": name, "start": active[1].strftime('%H:%M')}


def is_assigned_to_project(cur, emp_id, project_id):
    cur.execute("""
        SELECT 1 FROM "EmployeeProject"
        WHERE "employeeId" = %s AND "projectId" = %s AND "isDeleted" = FALSE
    """, (emp_id, project_id))
    return cur.fetchone() is not None


def parse_human_date(input_date: str):
    from datetime import datetime, timedelta

    try:
        input_date = input_date.strip().lower()
        today = datetime.today().date()

        if input_date == "today":
            return str(today), str(today)
        elif input_date == "yesterday":
            yday = today - timedelta(days=1)
            return str(yday), str(yday)
        elif "to" in input_date:
            try:
                start_raw, end_raw = input_date.split("to")
                start_date = datetime.strptime(start_raw.strip(), "%d %B").replace(year=today.year).date()
                end_date = datetime.strptime(end_raw.strip(), "%d %B").replace(year=today.year).date()
                return str(start_date), str(end_date)
            except:
                return "", ""
        else:
            try:
                single_date = datetime.strptime(input_date.strip(), "%d %B").replace(year=today.year).date()
                return str(single_date), str(single_date)
            except:
                try:
                    parsed = datetime.strptime(input_date.strip(), "%Y-%m-%d").date()
                    return str(parsed), str(parsed)
                except:
                    return "", ""
    except:
        return "", ""



# def get_timesheet_entries(cur, emp_id, start_date, end_date):
#     cur.execute("""
#         SELECT 
#             DATE(c."startTime")           AS date,
#             COALESCE(p."name", 'Unassigned') AS project,
#             c."startTime"                 AS in_dt,
#             c."endTime"                   AS out_dt,
#             c.duration                    AS dur,
#             c.status                      AS status,
#             c.id                          AS entry_id
#         FROM "ClockIn" c
#         LEFT JOIN "Project" p ON c."projectId" = p.id
#         WHERE c."employeeId" = %s
#           AND c."isDeleted" = FALSE
#           AND DATE(c."startTime") BETWEEN %s AND %s
#         ORDER BY c."startTime" DESC
#     """, (emp_id, start_date, end_date))
#     return cur.fetchall()


# def get_timesheet_entries(cur, emp_id, start_date=None, end_date=None, status=None):
#     sql = """
#     SELECT
#       DATE(c."startTime")       AS date,
#       COALESCE(p.name, 'Unassigned') AS project,
#       c."startTime"             AS in_dt,
#       c."endTime"               AS out_dt,
#       c.duration                AS dur,
#       c.status                  AS status,
#       c.id                      AS entry_id
#     FROM "ClockIn" c
#     LEFT JOIN "Project" p ON p.id = c."projectId"
#     WHERE c."employeeId" = %s
#       AND c."isDeleted" = FALSE
#     """
#     params = [emp_id]

#     # only add date filter if both provided
#     if start_date and end_date:
#         sql += ' AND DATE(c."startTime") BETWEEN %s AND %s'
#         params += [start_date, end_date]

#     # only add status filter if provided
#     if status:
#         sql += ' AND c.status = %s'
#         params.append(status.upper())

#     sql += ' ORDER BY c."startTime" DESC'

#     cur.execute(sql, params)
#     return cur.fetchall()


def get_timesheet_entries(cur, emp_id, start_date=None, end_date=None, status=None, project_name=None):
    sql = '''
    SELECT
      DATE(c."startTime")       AS date,
      COALESCE(p.name, 'Unassigned') AS project,
      c."startTime"             AS in_dt,
      c."endTime"               AS out_dt,
      c.duration                AS dur,
      c.status                  AS status,
      c.id                      AS entry_id
    FROM "ClockIn" c
    LEFT JOIN "Project" p ON p.id = c."projectId"
    WHERE c."employeeId" = %s
      AND c."isDeleted" = FALSE
    '''
    params = [emp_id]

    # Date filter: explicit range or default to last 7 days
    if start_date and end_date:
        sql += ' AND DATE(c."startTime") BETWEEN %s AND %s'
        params += [start_date, end_date]
    elif not start_date and not end_date:
        # Default to last 7 days
        sql += ' AND DATE(c."startTime") BETWEEN CURRENT_DATE - INTERVAL '"'6 days'"' AND CURRENT_DATE'

    # Status filter
    if status:
        sql += ' AND c.status = %s'
        params.append(status.upper())

    # Project name filter
    if project_name:
        sql += ' AND p.name ILIKE %s'
        params.append(f"%{project_name}%")

    sql += ' ORDER BY c."startTime" DESC'

    cur.execute(sql, params)
    return cur.fetchall()



# ----------------------------------------------------- helper functions ends here ------------------------------

def list_assigned_projects(state: AgentState) -> AgentState:
    try:
        emp_id = state.arguments.get("employeeId")
        with conn.cursor() as cur:
            projects = get_assigned_projects(cur, emp_id)
            if not projects:
                state.response = "⚠️ You have no assigned projects."
            else:
                project_list = "\n".join(f"- {p}" for p in projects)
                state.response = "📝 Your assigned projects:\n" + project_list
        return reset_state(state)
    except Exception as e:
        logger.exception("Failed to list assigned projects")
        state.response = f"❌ Could not fetch project list: {str(e)}"
        return reset_state(state)



def clock_in_to_project(state: AgentState) -> AgentState:
    try:
        args = state.arguments
        emp_id = args.get("employeeId")
        project_id = args.get("project_id")
        project_name = args.get("project_name")
        geolocation = args.get("geo_location")
        location_obj = None  # can be None
        if not emp_id:
            state.response = "❌ Missing employee ID."
            return state

        with conn.cursor() as cur:
            feature_enabled, geolocation_required = is_clock_in_allowed(cur, state.arguments.get("org_id"))

            if not feature_enabled:
                state.response = "❌ Clock-in is not enabled for your organization. Please contact your admin."
                return reset_state(state)

            if is_already_checked_in(cur, emp_id):
                project = get_active_project_name(cur, emp_id)
                state.response = f"⚠️ Already clocked in to **{project['name']}** since {project['start']} UTC."
                return reset_state(state)
            

                                    
            
            if project_name and not project_id:
                original_input = project_name
                project_id, project_name = get_project_id_by_name(cur, emp_id, original_input)
                if not project_id:
                    state.response = f"❌ You are not assigned to any project named '{project_name}'."
                    
                    return reset_state(state)

                state.arguments["project_id"] = project_id

            #  Handle dynamic user input as project name (if asked previously)
            if state.current_field == "project_name" and state.wait_for_input:
                user_input = state.user_input.strip()
                project_id, project_name = get_project_id_by_name(cur, emp_id, user_input)
                if project_id:
                    state.arguments["project_id"] = project_id
                    state.current_field = None
                    state.wait_for_input = False
                else:
                    state.response = f"❌ No assigned project found like '{user_input}'. Try again."
                    state.current_field = "project_name"
                    state.wait_for_input = True
                    return reset_state(state)


            # No project specified, show assigned project list
            if not project_id:
                project_list = get_assigned_projects(cur, emp_id)
                if not project_list:
                    state.response = "❌ You are not assigned to any active project."
                    return reset_state(state)

                state.response = (
                    "📝 You are not currently checked in.\n"
                    "Here are your assigned projects:\n"
                    + "\n".join(f"- {p}" for p in project_list)
                    + "\n👉 Please type the project name to clock in."
                )
                state.current_field = "project_name"
                state.wait_for_input = True
                state.function_name = "clock_in_to_project"
                return state


            

            #  Confirm project assignment
            if not is_assigned_to_project(cur, emp_id, project_id):
                state.response = "❌ You are not assigned to this project."
                return reset_state(state)
            

                        # Ask for geolocation if required and not yet provided
            if geolocation_required and not geolocation:
                state.response = "📍 Please provide your current location to clock in."
                state.current_field = "geo_location"
                state.tags = {"location_required": True}
                state.wait_for_input = True
                
                state.function_name = "clock_in_to_project"
                 # Add a geolocation tag
                
                return state
            
            # Step 1: Handle geo_location input when asked
            if state.current_field == "geo_location" and state.wait_for_input:
                user_input = state.user_input.strip()
                
                if user_input.lower() == "cancel":
                    state.response = "❌ Clock-in cancelled."
                    return reset_state(state)
                
                try:
                    lat_str, lng_str = map(str.strip, user_input.split(","))
                    location_obj = {
                        "lat": float(lat_str),
                        "lng": float(lng_str)
                    }
                    state.arguments["geo_location"] = location_obj  # ✅ Set parsed JSON here
                    state.current_field = None
                    state.wait_for_input = False
                    # ✅ Remove all geo-related tags
                    if isinstance(state.tags, dict):
                        for tag in ["location_required", "show_cancel"]:
                            state.tags.pop(tag, None)
                    return state
                except Exception as e:
                    logger.debug(f"Invalid geo_location input: {user_input} | Error: {e}")
                    state.response = (
                        "❌ Invalid location format. Use: `34.44, 35.445`\n"
                        "👉 Try again or type `cancel` to exit."
                    )
                    state.current_field = "geo_location"
                    state.wait_for_input = True
                    if not hasattr(state, "tags") or not isinstance(state.tags, dict):
                        state.tags = {}

                    state.tags.update({
                        "location_required": True,
                        "show_cancel": True
                    })
                    state.function_name = "clock_in_to_project"
                    return state

            state.tags = {}
            #  Insert check-in record
            now_utc = datetime.now(timezone.utc)

            cur.execute("""
                INSERT INTO "ClockIn" ("employeeId", "projectId", "startTime", "status", "createdAt", "updatedAt", "location")
                VALUES (%s, %s, %s, 'PENDING', now(), now(), %s)
                RETURNING id
            """, (emp_id, project_id, now_utc, json.dumps(location_obj)))
 
            entry_id = cur.fetchone()[0]
            conn.commit()

        state.response = f"✅ Clock-in to **{project_name or 'project'}** recorded at **{now_utc.strftime('%H:%M %p UTC')}**."
        return reset_state(state)

    except Exception as e:
        logger.exception("clock_in_to_project failed")
        state.response = f"❌ Failed to clock in: {str(e)}"
        return state



def clock_out_from_project(state: AgentState) -> AgentState:
    try:
        args = state.arguments
        emp_id = args.get("employeeId")

        if not emp_id:
            state.response = "❌ Missing employee ID."
            return state

        with conn.cursor() as cur:
            # ✅ Check if already clocked in today
            cur.execute("""
                SELECT id, "projectId", "startTime" FROM "ClockIn"
                WHERE "employeeId" = %s AND DATE("startTime") = CURRENT_DATE
                AND "endTime" IS NULL AND "isDeleted" = FALSE
            """, (emp_id,))
            active = cur.fetchone()

            if not active:
                state.response = "⚠️ You are not currently clocked in to any project today."
                return reset_state(state)

            clockin_id, project_id, start_time = active
            now_utc = datetime.now(timezone.utc)

            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)

            # Calculate duration in minutes
            duration = int((now_utc - start_time).total_seconds() // 60)

            # ✅ Update the record
            cur.execute("""
                UPDATE "ClockIn"
                SET "endTime" = %s, "duration" = %s, "updatedAt" = now()
                WHERE id = %s
            """, (now_utc, duration, clockin_id))
            conn.commit()

            cur.execute('SELECT name FROM "Project" WHERE id = %s', (project_id,))
            proj = cur.fetchone()
            project_name = proj[0] if proj else "your project"

        state.response = (
            f"✅ Clock-out from **{project_name}** successful.\n"
            f"🕐 Total time: **{duration} minutes** from {start_time.strftime('%H:%M')} to {now_utc.strftime('%H:%M')} UTC."
        )
        return reset_state(state)

    except Exception as e:
        logger.exception("clock_out_from_project failed")
        state.response = f"❌ Failed to clock out: {str(e)}"
        return state




# -------------------- Main Tool Function --------------------








def personal_timesheet(state: AgentState) -> AgentState:
    try:
        args          = state.arguments
        emp_id        = args.get("employeeId")
        employee_name   = args.get("employee_name")

        if not emp_id:
            state.response = "❌ Missing employee ID."
            return state

        # Extract possible filters
        project_name  = args.get("project_name")
        start         = args.get("start_date")
        end           = args.get("end_date")
        raw           = args.get("date")             # e.g. "7 July"
        status_filter = args.get("status")           # e.g. "PENDING" or "REJECTED"

        for k in ["start_date", "end_date"]:
            if args.get(k) == "null":
                args[k] = None

        # Parse raw human date
        if raw and not (start or end):
            start, end = parse_human_date(raw)
        # Mirror single-sided date
        if start and not end:
            end = start
        elif end and not start:
            start = end

        provided_date = bool(start or end)

        # Fetch rows (defaults to last 7 days if no dates provided)
        with conn.cursor() as cur:
            rows = get_timesheet_entries(cur, emp_id, start, end, status_filter, project_name)
        if not employee_name:
           with conn.cursor() as cur:
               cur.execute("SELECT name FROM \"Employee\" WHERE id = %s", (emp_id,))
               row = cur.fetchone()
               if row:
                 employee_name = row[0]
               else:
                 employee_name = "Unknown"

        # Handle no results
        if not rows:
            if provided_date:
                if start == end:
                    state.response = f"📭 No time entries found for {start}."
                else:
                    state.response = f"📭 No time entries found from {start} to {end}."
            else:
                state.response = "📭 No time entries found in the last 7 days."
            return reset_state(state)

        # Group and normalize
        grouped = defaultdict(list)
        for date_, proj, in_dt, out_dt, dur, status, eid in rows:
            if isinstance(in_dt, str):
                in_dt = dateutil_parser.parse(in_dt)
            if isinstance(out_dt, str) and out_dt:
                out_dt = dateutil_parser.parse(out_dt)

            date_key = date_.isoformat()
            grouped[date_key].append({
                "id":       eid,
                "project":  proj,
                "name":     employee_name,
                "clock_in": in_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "clock_out": out_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if out_dt else "",
                "duration": format_duration(dur),
                "status":   status,
                "editable": status in {"PENDING", "REJECTED"}
            })

        # Apply status filter on grouped entries
        if status_filter:
            sf = status_filter.strip().upper()
            for d in list(grouped):
                filtered = [e for e in grouped[d] if e["status"] == sf]
                if filtered:
                    grouped[d] = filtered
                else:
                    grouped.pop(d)

        # Build HTML table
        rows_html = ""
        for date_str, entries in grouped.items():
            for e in entries:
                rows_html += (
                    "<tr>"
                    f"<td>{date_str}</td>"
                    f"<td>{e['project']}</td>"
                    f"<td>{e['name']}</td>"
                    f"<td>{e['clock_in']}</td>"
                    f"<td>{e['clock_out'] or '—'}</td>"
                    f"<td>{e['duration']} min</td>"
                    f"<td>{e['status']}</td>"
                    "</tr>"
                )

        # Title line
        title_parts = ["📅 Timesheet"]
        if project_name:
            title_parts.append(f"for **{project_name}**")
        if provided_date:
            title_parts.append(f"from **{start}** to **{end}**")
        else:
            title_parts.append("(last 7 days)")
        title = " ".join(title_parts)

        table = (
            title
            + "<table>"
              "<thead><tr>"
                "<th>Date</th><th>Project</th><th>Clock-in</th>"
                "<th>Clock-out</th><th>Duration</th><th>Status</th>"  
              "</tr></thead><tbody>"
            + rows_html
            + "</tbody></table>"
        )

        state.metadata["timesheet_data"] = [
            {"date": date_str, "entries": entries}
            for date_str, entries in grouped.items()
        ]
        state.metadata["next_function_name"]= "update_timesheet_entry"
        state.tags["edit_timesheet"] = True
        state.response = f"Here's your {title}"
        return reset_state(state)

    except Exception as e:
        logger.exception("personal_timesheet failed")
        state.response = f"❌ Failed to fetch timesheet: {e}"
        return state





def parse_edit_timesheet_details(user_input: str) -> List[Dict]:
    user_input = user_input.lower()
    edits = []

    # Extract all occurrences of dates
    date_patterns = re.findall(r"(\d{1,2} \w+|\btoday\b|\byesterday\b)", user_input)
    clock_in_match = re.search(r"(clock[-\s]?in|start)\s*(to|as)?\s*(\d{1,2}[:.]\d{2})", user_input)
    clock_out_match = re.search(r"(clock[-\s]?out|end)\s*(to|as)?\s*(\d{1,2}[:.]\d{2})", user_input)
    project_match = re.search(r"for\s+([\w\s]+)", user_input)

    for d in date_patterns:
        edits.append({
            "date": d.strip(),
            "clock_in": clock_in_match.group(3) if clock_in_match else None,
            "clock_out": clock_out_match.group(3) if clock_out_match else None,
            "project_name": project_match.group(1).strip() if project_match else None
        })

    return edits

def fetch_entries_for_date(cur, emp_id, project_id, date):
    cur.execute("""
        SELECT id, TO_CHAR("startTime", 'HH24:MI') as start, 
                     TO_CHAR("endTime", 'HH24:MI') as end, 
                     "duration"
        FROM "ClockIn"
        WHERE "employeeId" = %s AND (%s IS NULL OR "projectId" = %s) 
        AND DATE("startTime") = %s AND "isDeleted" = FALSE
        ORDER BY "startTime"
    """, (emp_id, project_id,project_id, date))
    return [{"id": r[0], "start": r[1], "end": r[2], "duration": r[3]} for r in cur.fetchall()]


def format_entry_list(entries):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join([f"{letters[i]}. Entry ID {e['id']}, Clock-in: {e['start']}, Clock-out: {e['end'] or '—'}"
                      for i, e in enumerate(entries)])


def edit_timesheet_entry(state: AgentState) -> AgentState:
    try:
        args = state.arguments
        emp_id = args["employeeId"]
        project_name = args["project_name"]
        start_date, _ = parse_human_date(str(args["timesheet_date"]))
        clock_out = args.get("clock_out")
        clock_in = args.get("clock_in")

        if not emp_id:
            state.response = "❌ Missing employee ID."
            return state



        if not project_name or not start_date:
            state.response = "❌ Missing project name or date."
            return state


        # if not start_date:
        #     state.response = "❌ Invalid date format."
        #     return state

        with conn.cursor() as cur:
            project_id, project_name = get_project_id_by_name(cur, emp_id, project_name)
            if not project_id:
                state.response = f"❌ Project '{project_name}' not found or not assigned to you."
                return state

            matches = fetch_entries_for_date(cur, emp_id, project_id, start_date)
            if not matches:
                # No entry found → treat as missing entry and create a new one
                in_time = datetime.strptime(f"{start_date} {clock_in}", "%Y-%m-%d %H:%M") if clock_in else None
                out_time = datetime.strptime(f"{start_date} {clock_out}", "%Y-%m-%d %H:%M") if clock_out else None
                duration = int((out_time - in_time).total_seconds() // 60) if in_time and out_time else None

                cur.execute("""
                    INSERT INTO "ClockIn" ("employeeId", "projectId", "startTime", "endTime", "duration", "status", "createdAt", "updatedAt")
                    VALUES (%s, %s, %s, %s, %s, 'PENDING', now(), now())
                    RETURNING id
                """, (emp_id, project_id, in_time, out_time, duration))
                new_id = cur.fetchone()[0]
                conn.commit()

                state.response = f"🆕 Added new entry for **{project_name}** on {start_date}."
                return reset_state(state)

            if len(matches) > 1:
                # 🧠 Store and ask user to pick
                state.metadata["pending_edit_entries"] = matches
                info = {
                    "date": start_date,
                    "project_id": project_id,
                    "project_name": project_name
                }
                if clock_in:
                    info["clock_in"] = clock_in
                if clock_out:
                    info["clock_out"] = clock_out
                state.metadata["edit_info"] = info

                state.ask_user = (
                    f"🛠️ Multiple entries found for **{project_name}** on {start_date}:\n\n"
                    f"{format_entry_list(matches)}\n\n"
                    "Please reply with the entry letter (A, B, etc.) to edit."
                )
                state.wait_for_input = True
                return state

            # ✅ Single match, directly go to confirmation
            state.metadata["timesheet_edits"] = [{
                "id": matches[0]["id"],
                "project_name": project_name,
                "date": start_date,
                "clock_in": clock_in,
                "clock_out": clock_out
            }]
            state.ask_user = (
                f"🛠️ Ready to update clock-in/out for {project_name} on {start_date}. Should I proceed? (yes/no)"
            )
            state.tags["timezone"] = True
            state.validated["pending_confirmation"] = True

            state.wait_for_input = True
            return state

    except Exception as e:
        logger.exception("edit_timesheet_entry failed")
        state.response = f"❌ Error while editing timesheet: {str(e)}"
        return state








def update_timesheet_entry(state: AgentState) -> AgentState:
    try:
        emp_id = state.arguments.get("employeeId")
        timezone_str = state.arguments.get("timezone")

        edits = state.metadata.get("timesheet_edits") or state.arguments.get("edits")
        if not emp_id or not edits:
            state.response = "❌ Missing employee ID or update data."
            return state

        if all(not e.get("clock_in") and not e.get("clock_out") for e in edits):
            state.response = "❌ Please provide at least a new clock-in or clock-out time for update."
            return state
        
        try:
            user_tz = pytz.timezone(timezone_str)
        except Exception:
            state.response = f"❌ Invalid timezone: {timezone_str}"
            return state

        with conn.cursor() as cur:
            updated = []
            for e in edits:
                entry_id = e["id"]
                entry_date = e["date"] or e["timesheet_date"]  # Accept either key
                start_time, end_time = None, None
                project_name = e.get("project_name", "Unknown")

                # if e.get("clock_in"):
                #     start_time = datetime.strptime(f"{entry_date} {e['clock_in']}", "%Y-%m-%d %H:%M")
                # if e.get("clock_out"):
                #     end_time = datetime.strptime(f"{entry_date} {e['clock_out']}", "%Y-%m-%d %H:%M")

                if e.get("clock_in"):
                    local_dt = datetime.strptime(f"{entry_date} {e['clock_in']}", "%Y-%m-%d %H:%M")
                    localized_start = user_tz.localize(local_dt)
                    start_time = localized_start.astimezone(pytz.UTC)

                # Convert clock_out
                if e.get("clock_out"):
                    local_dt = datetime.strptime(f"{entry_date} {e['clock_out']}", "%Y-%m-%d %H:%M")
                    localized_end = user_tz.localize(local_dt)
                    end_time = localized_end.astimezone(pytz.UTC)

                if start_time and end_time and end_time <= start_time:
                    state.response = (
                        f"❌ Clock-out ({e['clock_out']}) must be after clock-in ({e['clock_in']})"
                    )
                    return reset_state(state)


                update_fields = []  
                values = []

                if start_time:
                    update_fields.append('"startTime" = %s')
                    values.append(start_time)
                if end_time:
                    update_fields.append('"endTime" = %s')
                    values.append(end_time)
                if start_time and end_time:
                    duration = int((end_time - start_time).total_seconds() // 60)
                    update_fields.append("duration = %s")
                    values.append(duration)

                values.append(entry_id)
                cur.execute(f"""
                    UPDATE "ClockIn" SET {', '.join(update_fields)}, "updatedAt" = now() WHERE id = %s
                """, tuple(values))
                updated.append(
                    f"✅ Updated {project_name} on {entry_date} "
                    f"{'(clock-in: ' + e['clock_in'] if e.get('clock_in') else ''}"
                    f"{', clock-out: ' + e['clock_out'] if e.get('clock_out') else ''})"
                )

            conn.commit()
            state.response = "📝 Update Summary:\n" + "\n".join(updated)
            state.metadata = {}
            state.tags = {}
            return reset_state(state)

    except Exception as e:
        logger.exception("update_timesheet_entry failed")
        state.response = f"❌ Failed to update timesheet: {str(e)}"
        return state

# def get_all_team_members(manager_id: int) -> List[int]:
#     """
#     Recursively fetch all employee IDs under a manager, including indirect reports.
#     """
#     team = set()
#     queue = [manager_id]
#     with conn.cursor() as cur:
#         while queue:
#             current_manager = queue.pop()
#             cur.execute(
#                 'SELECT id FROM "Employee" WHERE "managerId" = %s AND is_deleted = FALSE',
#                 (current_manager,)
#             )
#             reports = [r[0] for r in cur.fetchall()]
#             for emp in reports:
#                 if emp not in team:
#                     team.add(emp)
#                     queue.append(emp)
#     return list(team)
# # --- View handler ---
# def view_team_timesheet(state: AgentState) -> AgentState:
#     try:
#         # 1) Permission
#         if state.role not in ["manager", "hr"]:
#             state.response = "Only managers or HR can view team timesheets."
#             return reset_state(state)

#         args       = state.arguments
#         mgr_id     = args.get("employeeId")
#         org_id     = args.get("org_id")
#         team_of_employee = args.get("team_of_employee") or []

#         if isinstance(team_of_employee, bool):
#             team_of_employee = [team_of_employee] * len(employee_name or [])
#         # office_id  = args.get("officeId")
#         # 2) Date parsing / defaults
#         start = args.get("start_date")
#         end   = args.get("end_date")
#         raw   = args.get("date")
#         if raw and not (start or end):
#             start, end = parse_human_date(raw)
#         if start and not end:
#             end = start
#         elif end and not start:
#             start = end
#         if not (start and end):
#             today = datetime.now(timezone(timedelta(hours=5)) ).date()
#             end   = str(today)
#             start = str(today - timedelta(days=6))

#         # 3) Filters
#         status_filter = (args.get("status") or "PENDING").upper()
#         employee_name   = args.get("employee_name")
#         project_name    = args.get("project_name")
#         # team = get_all_team_members(mgr_id)

#         # 4) Fetch team member IDs
#         with conn.cursor() as cur:
#             cur.execute(
#                 'SELECT id FROM "Employee" '
#                 'WHERE "managerId"=%s AND is_deleted=FALSE',
#                 (mgr_id,)
#             )
#             team = [r[0] for r in cur.fetchall()]
#         if not team:
#             state.response = "You have no team members assigned."
#             return reset_state(state)
#         # filter by employee_name if provided
#         if employee_name:
#             # if isinstance(employee_name, str):
#             #     employee_name = [employee_name]
#             matched = []
#             with conn.cursor() as cur:
#                 for name in employee_name:
#                     cur.execute(
#                     'SELECT id FROM "Employee" WHERE "managerId"=%s  AND name ILIKE %s',
#                     (mgr_id, f"%{name}%")
#                     )
#                     matched += [r[0] for r in cur.fetchall()]
#             matched = list(set(matched))  # remove duplicates
#             if not matched:
#                 state.response = f"No team member found matching '{employee_name}'."
#                 return state
#             team = matched







#         # 5) Build SELECT
#         placeholders = ",".join(["%s"]*len(team))
#         sql = f"""
#             SELECT
#                 c.id, c."employeeId", e.name, c."projectId", p.name AS project_name,
#                 c."startTime", c."endTime", c.duration, c.status
#             FROM "ClockIn" c
#             JOIN "Employee" e ON e.id = c."employeeId"
#             LEFT JOIN "Project" p ON p.id = c."projectId"
#             WHERE c."employeeId" IN ({placeholders})
#               AND c.status = %s
#               AND DATE(c."startTime") BETWEEN %s AND %s
#         """
#         params = team + [status_filter, start, end]
#         if project_name:
#             sql += " AND p.name ILIKE %s"
#             params.append(f"%{project_name}%")
#         sql += ' ORDER BY c."startTime" DESC'

#         # 6) Execute + render
#         with conn.cursor() as cur:
#             cur.execute(sql, params)
#             rows = cur.fetchall()
#         if not rows:
#             state.response = f"No {status_filter.lower()} entries between {start} and {end}."
#             return reset_state(state)

#         rows_html = ""
#         meta      = []
#         for eid, emp, ename, pid, pname, in_dt, out_dt, dur, st in rows:
#             rows_html += (
#                 "<tr>"
#                 f"<td><input type='checkbox' value='{eid}'></td>"
#                 f"<td>{ename}</td>"
#                 f"<td>{pname or '—'}</td>"
#                 f"<td>{in_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}</td>"
#                 f"<td>{out_dt.strftime('%Y-%m-%dT%H:%M:%SZ') if out_dt else '—'}</td>"
#                 f"<td>{format_duration(dur)}</td>"
#                 f"<td>{st}</td>"
#                 "</tr>"
#             )
#             meta.append({
#                 "entry_id":   eid,
#                 "employeeName": ename,
#                 "employeeId": emp,
#                 "projectname": pname,
#                 "projectId":  pid,
#                 "startTime":  in_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
#                 "endTime":    out_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if out_dt else None,
#                 "duration":   format_duration(dur),
#                 "status":     st
#             })


#         state.metadata["team_timesheet_data"] = meta
#         state.metadata["function_name"] = "update_team_timesheet"

        
#         pdf_link = build_team_timesheet_pdf(rows, start, end, status_filter)
#         state.metadata["pdf_link"] = pdf_link 
#         rows_html += (
#             "<tr><td colspan='7' style='text-align: right; padding-top: 10px;'>"
#             f"📎 <a href='{pdf_link}' target='_blank'>Download PDF</a>"
#             "</td></tr>"
#         )
#         if not hasattr(state, "tags") or not isinstance(state.tags, dict):
#             state.tags = {}

#         state.tags["view_team_timsheet"] = True
#         # Final response with header + table
#         state.response = (
#             f"📋 Team timesheets ({status_filter.lower()}) from {start} to {end}"
#             )

#         return reset_state(state)

#     except Exception as e:
#         logger.exception("view_team_timesheet failed")
#         state.response = f"❌ Failed to fetch team timesheets: {e}"
#         return state

def get_all_team_members(manager_id: int) -> List[int]:
    """
    Recursively fetch all employee IDs under a manager, including indirect reports.
    """
    team = set()
    queue = [manager_id]
    with conn.cursor() as cur:
        while queue:
            current_manager = queue.pop()
            cur.execute(
                'SELECT id FROM "Employee" WHERE "managerId" = %s AND is_deleted = FALSE',
                (current_manager,)
            )
            reports = [r[0] for r in cur.fetchall()]
            for emp in reports:
                if emp != manager_id and emp not in team:
                    team.add(emp)
                    queue.append(emp)
    return list(team)
 
 
# --- View handler ---
def view_team_timesheet(state: AgentState) -> AgentState:
    try:
        if state.role not in ["manager", "hr"]:
            state.response = "Only managers or HR can view team timesheets."
            return reset_state(state)
 
        args = state.arguments
        mgr_id = args.get("employeeId")
        org_id = args.get("org_id")
        employee_name = args.get("employee_name")
        project_name = args.get("project_name")
        view_sub_team = args.get("view_sub_team", True)
        if isinstance(employee_name, str):
            # Split into a list if commas exist
            employee_name = [name.strip() for name in employee_name.split(",") if name.strip()]
        elif not isinstance(employee_name, list):
            employee_name = []

        # Remove empty/None entries from the list
        employee_name = [name for name in employee_name if name and name.strip()]

        if not employee_name:
            view_sub_team = False


        # 1. Parse date range
        start = args.get("start_date")
        end = args.get("end_date")
        raw = args.get("date")
        if raw and not (start or end):
            start, end = parse_human_date(raw)
        if start and not end:
            end = start
        elif end and not start:
            start = end

        today = datetime.now(timezone(timedelta(hours=5))).date()    
        if not (start and end):
            
            end = str(today)
            start = str(today - timedelta(days=6))

        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()

        # Cap to today if LLM gave future dates
        if start_dt > today:
            start_dt = today
        if end_dt > today:
            end_dt = today

        # Convert back to string
        start = str(start_dt)
        end = str(end_dt)

 
        status_filter_raw = args.get("status") or "PENDING"
        if isinstance(status_filter_raw, list):
            status_filter = [s.upper() for s in status_filter_raw]
            status_text = ", ".join(s.lower() for s in status_filter)
        else:
            status_filter = [status_filter_raw.upper()]
            status_text = status_filter_raw.lower()


 
        # 2. Determine whose team we are viewing
        target_manager_id = mgr_id  # default to current logged-in manager
 
        team = []

        if employee_name:
            matched_ids = []
            with conn.cursor() as cur:
                for name in employee_name:
                    cur.execute(
                        '''
                        SELECT e.id
                        FROM "Employee" e
                        JOIN "Office" o ON e."officeId" = o.id
                        WHERE e.name ILIKE %s
                        AND e.is_deleted = FALSE
                        AND o."organizationId" = %s
                        ''',
                        (f"%{name}%", org_id)
                    )
                    rows = cur.fetchall()
                    matched_ids.extend([r[0] for r in rows])

            if not matched_ids:
                state.response = f"No employee(s) found with name(s): {', '.join(employee_name)}."
                return reset_state(state)

            for emp_id in matched_ids:
                if view_sub_team:
                    state.tags["view_sub_team"] = True
                    sub_team = get_all_team_members(emp_id)
                    if sub_team:
                        team.extend(sub_team)
                    else:
                        logger.info(f"Employee ID {emp_id} has no sub-team.")
                        if not view_sub_team:
                            team.append(emp_id)

                else:
                    team.append(emp_id)
            team = list(set(team))
        else:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT id FROM "Employee" WHERE "managerId" = %s AND is_deleted = FALSE',
                    (mgr_id,)
                )
                team = [r[0] for r in cur.fetchall()]

            if not team:
                state.response = f"⚠️ You currently have no team members reporting to you."
                return reset_state(state)
                
        
        if not team:
            logger.warning(f"No team found for request: employee_name={employee_name}, mgr_id={mgr_id}")
            state.response = "⚠️ No team members found for the requested query."
            return reset_state(state)
 
        # 4. Build SQL query
        placeholders = ",".join(["%s"] * len(team))
        sql = f"""
            SELECT
                c.id, c."employeeId", e.name, c."projectId", p.name AS project_name,
                c."startTime", c."endTime", c.duration, c.status
            FROM "ClockIn" c
            JOIN "Employee" e ON e.id = c."employeeId"
            LEFT JOIN "Project" p ON p.id = c."projectId"
            WHERE c."employeeId" IN ({placeholders})
              AND c.status IN ({','.join(['%s'] * len(status_filter))})
              AND DATE(c."startTime") BETWEEN %s AND %s
        """
        params = team + status_filter + [start, end]
        if project_name:
            sql += " AND p.name ILIKE %s"
            params.append(f"%{project_name}%")
        sql += ' ORDER BY c."startTime" DESC'
 
        # 5. Execute query
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
 
        if not rows:
            state.response = f"No ({status_text}) entries between {start} and {end}."
            return reset_state(state)
 
        # 6. Build table
        rows_html = ""
        meta = []
        for eid, emp, ename, pid, pname, in_dt, out_dt, dur, st in rows:
            rows_html += (
                "<tr>"
                f"<td><input type='checkbox' value='{eid}'></td>"
                f"<td>{ename}</td>"
                f"<td>{pname or '—'}</td>"
                f"<td>{in_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}</td>"
                f"<td>{out_dt.strftime('%Y-%m-%dT%H:%M:%SZ') if out_dt else '—'}</td>"
                f"<td>{format_duration(dur)}</td>"
                f"<td>{st}</td>"
                "</tr>"
            )
            meta.append({
                "entry_id": eid,
                "employeeName": ename,
                "employeeId": emp,
                "projectname": pname,
                "projectId": pid,
                "startTime": in_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endTime": out_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if out_dt else None,
                "duration": format_duration(dur),
                "status": st,
                "editable": st.upper() != "APPROVED"
                
            })
 
        # 7. Finalize metadata + response
        state.metadata["team_timesheet_data"] = meta
        state.metadata["function_name"] = "update_team_timesheet"
 
        pdf_link = build_team_timesheet_pdf(rows, start, end, status_filter)
        state.metadata["pdf_link"] = pdf_link
        rows_html += (
            "<tr><td colspan='7' style='text-align: right; padding-top: 10px;'>"
            f"📎 <a href='{pdf_link}' target='_blank'>Download PDF</a>"
            "</td></tr>"
        )
 
        if not hasattr(state, "tags") or not isinstance(state.tags, dict):
            state.tags = {}
 
        state.tags["view_team_timsheet"] = True
        state.response = f"📋 Team timesheets ({status_text}) from {start} to {end}"
        return reset_state(state)
 
    except Exception as e:
        logger.exception("view_team_timesheet failed")
        state.response = f"❌ Failed to fetch team timesheets: {e}"
        return state

# --- Update handler ---

def safe_strip_upper(val):
    return val.strip().upper() if isinstance(val, str) else ""

def safe_strip(val):
    return val.strip() if isinstance(val, str) else ""



def update_team_timesheet(state: AgentState) -> AgentState:
    try:
        # 1) Permissions
        if state.role not in ["manager", "hr"]:
            state.response = "Only managers or HR can update timesheets."
            return state

        args          = state.arguments
        reason        = args.get("reason")
        status_raw = safe_strip(args.get("status"))
        filter_status = safe_strip_upper(args.get("filter_status"))


        # 1a) Normalize status synonyms
        status_synonyms = {
            "approve": "APPROVED",
            "approved": "APPROVED",
            "reject": "REJECTED",
            "rejected": "REJECTED",
            "deny": "REJECTED",
            "decline": "REJECTED",
            "pending": "PENDING",
            "wait": "PENDING"
        }
        status = status_synonyms.get(status_raw.lower(), status_raw.upper())
        valid_statuses = {"PENDING", "APPROVED", "REJECTED"}

        if status not in valid_statuses:
            state.response = f"❌ Invalid status '{status_raw}'. Valid statuses: {', '.join(valid_statuses)}."
            return state

        # 1b) If rejecting & no reason yet, prompt for it
        if status == "REJECTED" and not reason and not state.wait_for_input:
            state.ask_user       = "Please provide a reason for rejecting these timesheets:"
            state.current_field  = "reason"
            state.wait_for_input = True
            return state

        # 2) Extract all filters
        entries   = args.get("entries")
        raw       = args.get("date")
        start     = args.get("start_date")
        end       = args.get("end_date")
        emp_name  = args.get("employee_name")
        proj_name = args.get("project_name")

        # 2a) Normalize natural‐language → ISO dates
        if raw and not (start or end):
            start, end = parse_human_date(raw)
        if start and not end:
            end = start
        elif end and not start:
            start = end

        if start and end:
            try:
                sd = datetime.fromisoformat(start).date()
                ed = datetime.fromisoformat(end).date()
                if ed < sd:
                    state.response = "❌ End date must be on or after start date."
                    return state
            except ValueError:
                state.response = "❌ Dates must be YYYY-MM-DD or natural language like 'today'."
                return state

        # 3) Fetch your team member IDs
        mgr_id = args.get("employeeId")
        with conn.cursor() as cur:
            cur.execute(
                'SELECT id FROM "Employee" WHERE "managerId" = %s AND is_deleted = FALSE',
                (mgr_id,)
            )
            team_ids = [r[0] for r in cur.fetchall()]
        if not team_ids:
            state.response = "⚠️ You have no team members to update."
            return state

        # 4) Build dynamic UPDATE
        sql           = 'UPDATE "ClockIn" SET status=%s'
        params        = [status]
        where_clauses = []

        if status == "REJECTED":
            sql += ', reason=%s'
            params.append(reason)

        sql += ', "updatedAt" = now()'

        # 4a) Never touch already-APPROVED rows
        where_clauses.append('status != %s')
        params.append('APPROVED')

        # 4b) Always restrict to your team
        team_ph = ",".join(["%s"] * len(team_ids))
        where_clauses.append(f'"employeeId" IN ({team_ph})')
        params += team_ids

        # 4c) User-supplied filters
        if entries:
            ph = ",".join(["%s"] * len(entries))
            where_clauses.append(f"id IN ({ph})")
            params += entries
        else:
            if start and end:
                where_clauses.append('DATE("startTime") BETWEEN %s AND %s')
                params += [start, end]
            if emp_name:
                where_clauses.append(
                    '"employeeId" IN (SELECT id FROM "Employee" WHERE name ILIKE %s)'
                )
                params.append(f"%{emp_name}%")
            if proj_name:
                where_clauses.append(
                    '"projectId" IN (SELECT id FROM "Project" WHERE name ILIKE %s)'
                )
                params.append(f"%{proj_name}%")

            # 4d) Filter by prior status (e.g., "approve pending")
            if filter_status in {"PENDING", "REJECTED"}:
                where_clauses.append('status = %s')
                params.append(filter_status)
            elif status == "APPROVED" and not filter_status:
                # Default to filtering pending entries if not explicitly given
                where_clauses.append('status = %s')
                params.append("PENDING")

        # 5) Finalize & execute
        sql += " WHERE " + " AND ".join(where_clauses)
        sql += " RETURNING id"

        with conn.cursor() as cur:
            cur.execute(sql, params)
            updated = [r[0] for r in cur.fetchall()]
            if updated:
                cur.execute(
                    f'''
                    SELECT c.id, e.id AS emp_id, e.name, e."fcmToken", e."officeId", p.name AS project_name
                    FROM "ClockIn" c
                    JOIN "Employee" e ON c."employeeId" = e.id
                    LEFT JOIN "Project" p ON c."projectId" = p.id
                    WHERE c.id IN ({",".join(["%s"] * len(updated))})
                    ''',
                    updated
                )
                updates = []
                for row in cur.fetchall():
                    timesheet_id, emp_id, emp_name, fcm_token, office_id, project_name = row
                    updates.append(f"{emp_name} - {project_name or 'N/A'}")
                    notif_title = "Timesheet Update"
                    notif_body = f"Your timesheet for project '{project_name or 'N/A'}' has been {status.lower()}."

                    if fcm_token:
                        send_push_notification(
                            fcm_token=fcm_token,
                            title=notif_title,
                            body=notif_body
                        )

                    cur.execute(
                        """INSERT INTO "Notification" 
                        ("senderId", "recipientId", "officeId", "notification_type", "title", "message", "status", "is_deleted", "created_at")
                        VALUES (%s, %s, %s, %s, %s, %s, 'unread', FALSE, now())""",
                        (
                            mgr_id,
                            emp_id,
                            office_id,
                            "timesheet_update",
                            notif_title,
                            notif_body
                        )
                    )

            conn.commit()
        state.metadata = {}
        state.tags = {}
        # 6) Build response
        if not updated:
            state.response = "📭 No timesheets matched your request — everything may already be approved or there's nothing pending right now."
        else:
            entry_summary = """
                <table>
                    <thead>
                        <tr>
                            <th>Employee Name</th>
                            <th>Project Name</th>
                            <th>Date</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                """

                # For each updated row, fetch the date from ClockIn (startTime)
            with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT c.id, e.name, p.name AS project_name, DATE(c."startTime") AS entry_date, c.status
                        FROM "ClockIn" c
                        JOIN "Employee" e ON c."employeeId" = e.id
                        LEFT JOIN "Project" p ON c."projectId" = p.id
                        WHERE c.id IN ({','.join(['%s'] * len(updated))})
                        """,
                        updated
                    )
                    for row in cur.fetchall():
                        _, name, project, date, status_value = row
                        entry_summary += f"""
                            <tr>
                                <td>{name}</td>
                                <td>{project or '—'}</td>
                                <td>{date}</td>
                                <td>{status_value}</td>
                            </tr>
                        """

            entry_summary += "</tbody></table>"
            state.response = entry_summary


        return reset_state(state)

    except Exception as e:
        logger.exception("update_team_timesheet failed")
        state.response = f"Failed to update timesheets: {e}"
        return state


