import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ITSM ML Dashboard",
    layout="wide"
)

st.title("🛠️ ITSM Machine Learning Dashboard")

# ======================================================
# SIDEBAR INPUTS
# ======================================================

st.sidebar.header("📌 Ticket Details")

CI_Cat = st.sidebar.selectbox(
    "CI Category",
    ['subapplication', 'application', 'computer', 'displaydevice', 'software',
 'storage', 'database', 'hardware', 'officeelectronics', 'networkcomponents',
 'applicationcomponent', 'Phone']
)

CI_Subcat = st.sidebar.selectbox(
    "CI Subcategory",
    ['Web Based Application', 'Server Based Application', 'Desktop Application',
 'SAP', 'Client Based Application', 'Citrix', 'Standard Application',
 'Windows Server', 'Laptop', 'Linux Server', 'Monitor', 'Automation Software',
 'SAN', 'Banking Device', 'Desktop', 'Database', 'Oracle Server', 'Keyboard',
 'Printer', 'Exchange', 'System Software', 'VDI', 'Encryption', 'Omgeving',
 'MigratieDummy', 'Scanner', 'Controller', 'DataCenterEquipment',
 'KVM Switches', 'Switch', 'Database Software', 'Network Component',
 'Unix Server', 'Lines', 'ESX Cluster', 'zOS Server', 'SharePoint Farm',
 'NonStop Server', 'Application Server', 'Security Software', 'Thin Client',
 'Router', 'Net Device', 'Neoview Server', 'MQ Queue Manager', 'zOS Cluster',
 'UPS', 'Number', 'Iptelephony', 'Windows Server in extern beheer', 'Modem',
 'X86 Server', 'ESX Server', 'Virtual Tape Server', 'IPtelephony',
 'NonStop Harddisk', 'RAC Service', 'zOS Systeem', 'Firewall', 'Instance',
 'Protocol', 'Tape Library']
)

Category = st.sidebar.selectbox(
    "Ticket Category",
    [
        "incident",
        "request for information",
        "complaint",
        "request for change"
    ]
)

Impact = st.sidebar.slider("Impact", 1, 5, 3)
Urgency = st.sidebar.slider("Urgency", 1, 5, 3)

No_of_Reassignments = st.sidebar.number_input("Reassignments", 0, 20, 0)
No_of_Related_Incidents = st.sidebar.number_input("Related Incidents", 0, 50, 0)
No_of_Related_Interactions = st.sidebar.number_input("Related Interactions", 0, 50, 0)
No_of_Related_Changes = st.sidebar.number_input("Related Changes", 0, 50, 0)

Ticket_Short_Description = st.sidebar.text_area(
    "Short Description of Ticket in 500 words",
    ""
)

payload = {
    "CI_Cat": CI_Cat,
    "CI_Subcat": CI_Subcat,
    "Category": Category,
    "Impact": Impact,
    "Urgency": Urgency,
    "No_of_Reassignments": No_of_Reassignments,
    "No_of_Related_Incidents": No_of_Related_Incidents,
    "No_of_Related_Interactions": No_of_Related_Interactions,
    "No_of_Related_Changes": No_of_Related_Changes,
    "Ticket_Short_Description": Ticket_Short_Description
}

# ======================================================
# TABS
# ======================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Ticket Priority Classifier ", "🏢 Auto-tagging Tickets", "🔁 RFC failure Prediction", "📈 Forecast Incidents Occurance"]
)

# ======================================================
# PRIORITY
# ======================================================

with tab1:
    st.subheader("🎯 Ticket Priority Prediction")

    if st.button("Predict Priority"):
        res = requests.post(f"{API_URL}/predict_Level_of_priority", json=payload)
        if res.status_code == 200:
            st.success(res.json()["Level_of_priority"])
            st.json(res.json())
        else:
            st.error("Prediction failed")

# ======================================================
# PRIORITY + DEPARTMENT
# ======================================================

with tab2:
    st.subheader("🏢 Auto-tagging Tickets")

    if st.button("Predict Priority & Department"):
        res = requests.post(f"{API_URL}/ticket_priority_and_department", json=payload)
        if res.status_code == 200:
            st.success("Prediction Successful")
            st.json(res.json())
        else:
            st.error("Prediction failed")

# ======================================================
# RFC
# ======================================================

with tab3:
    st.subheader("🔁 RFC failure Prediction")

    if st.button("Predict RFC"):
        res = requests.post(f"{API_URL}/predict_rfc", json=payload)
        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error("RFC prediction failed")

# ======================================================
# FORECAST
# ======================================================

with tab4:
    st.subheader("📈 Incident Occurance Forecasting")

    model_name = st.selectbox(
        "Select Model",
        [
            "Overall Incident Forecast",
            "Application Incidents",
            "Sub-Application Incidents",
            "Server Based Application Incidents",
            "Web Based Application Incidents"
        ]
    )

    steps = st.slider("Forecast Months", 3, 36, 12)

    if st.button("Generate Forecast"):
        res = requests.post(
            f"{API_URL}/forecast",
            json={"model_name": model_name, "steps": steps}
        )

        if res.status_code == 200:
            data = res.json()

            df = pd.DataFrame.from_dict(
                data["monthly"], orient="index", columns=["Incidents"]
            )
            df.index = pd.to_datetime(df.index)

            st.line_chart(df)
            st.dataframe(df)

        else:
            st.error("Forecast failed")
