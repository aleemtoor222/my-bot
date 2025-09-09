# your_llm_tool_router.py
from typing import List, Dict
from llmconfig import llm
import json, re


import re, json
from typing import List, Dict
from llmconfig import llm

def detect_function_calls(user_input: str) -> List[Dict]:
    system_prompt = f"""
You are a smart HR assistant. From a user's message, extract 1 or more function calls.

Wrap your JSON in <JSON>...</JSON> and include:
- function_name: name of the tool (e.g. apply_leave, query_policy)
- arguments: dictionary of any arguments (e.g. from_date, query, leave_type)

User Input:
"{user_input}"

Return like this:

<JSON>
[
  {{
    "function_name": "apply_leave",
    "arguments": {{"leave_type":"sick","from_date":"2025-05-01","to_date":"2025-05-02"}}
  }},
  {{
    "function_name": "query_policy",
    "arguments": {{"query": "how many leaves i have"}}
  }}
]
</JSON>

Only include known functions: apply_leave, apply_travel, apply_reimbursement, query_policy, view_timesheet, edit_timesheet_entry, clock_in_to_project, clock_out_from_project, greetings, show_team, show_team_member_leaves, get_attendance, get_team_attendance, get_manager_detail

Functions and their possible arguments:
- apply_leave: leave_type, from_date, to_date, reason
- apply_travel: travel_type, city, country, from_date, to_date, purpose_of_travel, mode_of_transport
- apply_reimbursement: (let it be open)
- query_policy: query
- show_team: employee_name, employeeId
- show_team_member_leaves: employee_name
- get_attendance: email, employeeId
- get_team_attendance: startDate, endDate, personName
- get_manager_detail: employee_name, employeeId
- clock_in_to_project: project_name
- clock_out_from_project: none
- view_timesheet: start_date, end_date
- greetings: none
- edit_timesheet_entry: project_name, timesheet_date, clock_in, clock_out
"""

    try:
        raw = llm.invoke(system_prompt)
        match = re.search(r"<JSON>(.*?)</JSON>", raw, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return parsed
            except Exception as e:
                print(f"[JSON Parse Error]: {e}")
                return []
        else:
            print("[Regex Error]: <JSON> block not found.")
            return []
    except Exception as e:
        print(f"[Router Parse Error]: {e}")
        return [{"function_name": "clarify_intent", "arguments": {}}]


def format_multi_tool_response(user_input: str, tool_outputs: str) -> str:
    prompt = f"""
You are a helpful HR assistant.

The user said:
"{user_input}"

These tools were executed:
{tool_outputs}

Now respond to the user in a clean, conversational way using simple and helpful language.
Use emojis where helpful. Wrap tabular results in <table> tags. Be professional and polite.
"""
    return llm.invoke(prompt).strip()
