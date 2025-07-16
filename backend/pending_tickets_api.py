import os
from flask import Blueprint, jsonify, request
import requests
from requests.auth import HTTPBasicAuth

pending_tickets_api = Blueprint('pending_tickets_api', __name__)

# --- Close ticket endpoint ---
@pending_tickets_api.route('/api/tickets/<ticket_id>/close', methods=['POST'])
def close_ticket(ticket_id):
    """
    Close a ServiceNow ticket (incident) by setting its state to 7 (Closed).
    """
    url = f"https://{INSTANCE}.service-now.com/api/now/table/incident/{ticket_id}"
    payload = {"state": "7"}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        resp = requests.patch(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), headers=headers, json=payload, timeout=20)
        if resp.status_code in (200, 204):
            return jsonify({"success": True})
        else:
            return jsonify({"error": resp.text}), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500
import os
from flask import Blueprint, jsonify, request
import requests
from requests.auth import HTTPBasicAuth

pending_tickets_api = Blueprint('pending_tickets_api', __name__)

# --- Health check endpoint ---
@pending_tickets_api.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

# === Update these or use environment variables for security ===
INSTANCE = os.environ.get("SN_INSTANCE", "wiprodemo4")
USERNAME = os.environ.get("SN_USERNAME", "jagadeeswar.v@wipro.com")
PASSWORD = os.environ.get("SN_PASSWORD", "Jagadeeswar@01")
YOUR_EMAIL = os.environ.get("SN_EMAIL", "jagadeeswar.v@wipro.com")

# === Incident states to include (Pending only) ===
PENDING_STATES = ["1", "2", "3"]

@pending_tickets_api.route("/api/pending-tickets", methods=["GET"])
def get_pending_tickets():
    url = f"https://{INSTANCE}.service-now.com/api/now/table/incident"
    query = f"caller_id.email={YOUR_EMAIL}^stateIN{','.join(PENDING_STATES)}"
    params = {
        "sysparm_query": query,
        # Add all fields needed for frontend mapping
        "sysparm_fields": "number,short_description,state,urgency,assigned_to,opened_at,priority,category,u_category,category_name,cmdb_ci,sys_created_on,created_on,priority_label,severity",
        "sysparm_display_value": "true",  # Get display values for reference fields
        "sysparm_limit": request.args.get("limit", "10")
    }
    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(USERNAME, PASSWORD),
            headers={"Accept": "application/json"},
            params=params
        )
        state_map = {
            "1": "Pending",  # Changed from "New" to "Pending" for frontend compatibility
            "2": "In Progress",
            "3": "On Hold",
            "6": "Resolved",
            "7": "Closed"
        }
        urgency_map = {
            "1": "Critical",
            "2": "High",
            "3": "Moderate"
        }
        if response.status_code == 200:
            data = response.json()
            tickets = []
            for ticket in data.get("result", []):
                # Fix urgency mapping: if already a string (e.g., 'High'), use as is; else map or fallback to raw value
                raw_urgency = ticket.get("urgency")
                urgency_val = urgency_map.get(raw_urgency)
                if not urgency_val and isinstance(raw_urgency, str) and raw_urgency not in urgency_map:
                    urgency_val = raw_urgency  # Use as is if not mapped
                if not urgency_val:
                    urgency_val = "Unknown"
                tickets.append({
                    "id": ticket.get("number"),
                    "description": ticket.get("short_description"),
                    "status": state_map.get(ticket.get("state"), ticket.get("state", "")).lower().replace(" ", ""),
                    "urgency": urgency_val,
                    "state": state_map.get(ticket.get("state"), ticket.get("state", "")),
                    "assigned_to": ticket.get("assigned_to"),
                    "opened_at": ticket.get("opened_at"),
                    "priority": ticket.get("priority"),
                    "priority_label": ticket.get("priority_label"),
                    "severity": ticket.get("severity"),
                    "category": ticket.get("category"),
                    "u_category": ticket.get("u_category"),
                    "category_name": ticket.get("category_name"),
                    "cmdb_ci": ticket.get("cmdb_ci"),
                    "sys_created_on": ticket.get("sys_created_on"),
                    "created_on": ticket.get("created_on"),
                })
            return jsonify({"tickets": tickets})
        else:
            return jsonify({"error": response.text}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500
