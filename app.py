"""OSU P-card audit app.

Tab 1: auditors ask questions in natural language. Gemini (called from the backend)
translates the question into one read-only SQLite query, which the app validates and runs.
Tab 2: dashboard for keyword searches in the Description or Vendor field to find
transactions related to prohibited purchases. Uses fixed, parameterised SQL (no LLM call).

The Gemini API key is never stored in the code. It is read from Streamlit secrets
(.streamlit/secrets.toml locally, the Secrets editor on Streamlit Community Cloud)
or from a local .env file.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "pcards.db"
OSU_AGENCY_NUMBER = 1000
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_DISPLAY_ROWS = 5000
QUERY_TIMEOUT_SECONDS = 20

# ISO date (YYYY-MM-DD) built from the text TransactionDate ("M/D/YYYY 0:00:00"),
# so that results sort chronologically instead of alphabetically.
ISO_DATE_SQL = (
    "printf('%04d-%02d-%02d', Year, Month, "
    "CAST(substr(TransactionDate, instr(TransactionDate, '/') + 1, "
    "instr(substr(TransactionDate, instr(TransactionDate, '/') + 1), '/') - 1) AS INTEGER))"
)

load_dotenv()  # reads a local .env file if present (never committed to Git)

st.set_page_config(page_title="OSU P-card Audit", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
def get_setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from Streamlit secrets first, then from environment variables (.env)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml available
        pass
    return os.getenv(name, default)


def gemini_secret() -> str | None:
    return get_setting("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Database access (read-only, OSU population only)
# ---------------------------------------------------------------------------
def _authorizer(action, arg1, arg2, db_name, source):
    """Allow only reading the pcards view. Everything else (writes, PRAGMA, ATTACH,
    other tables, direct access to main.pcards) is denied."""
    if action == sqlite3.SQLITE_READ:
        if arg1 == "pcards" and (db_name == "temp" or source == "pcards"):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def get_connection() -> sqlite3.Connection:
    """Open the database read-only. A temporary view named pcards shadows the real table
    and restricts every query to the OSU population (AgencyNumber = 1000)."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute(
        f"CREATE TEMP VIEW pcards AS SELECT * FROM main.pcards "
        f"WHERE AgencyNumber = {OSU_AGENCY_NUMBER}"
    )
    conn.set_authorizer(_authorizer)
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    return conn


def run_query(sql: str, params: tuple = ()) -> tuple[pd.DataFrame, bool]:
    """Run a SELECT and return (results, truncated)."""
    conn = get_connection()
    try:
        cur = conn.execute(sql, params)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_DISPLAY_ROWS + 1)
    finally:
        conn.close()
    truncated = len(rows) > MAX_DISPLAY_ROWS
    return pd.DataFrame(rows[:MAX_DISPLAY_ROWS], columns=columns), truncated


@st.cache_data
def available_years() -> list[int]:
    df, _ = run_query("SELECT DISTINCT Year FROM pcards ORDER BY Year DESC")
    return df["Year"].tolist()


@st.cache_data
def schema_text() -> str:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cols = conn.execute("PRAGMA table_info(pcards)").fetchall()
    finally:
        conn.close()
    return ", ".join(f"{c[1]} {c[2]}" for c in cols)


# ---------------------------------------------------------------------------
# Tab 1: natural language to SQL
# ---------------------------------------------------------------------------
def build_prompt(question: str, selected_year: int) -> str:
    return f"""You translate auditor questions into one safe SQLite query. The only approved table is:
pcards({schema_text()})

The app automatically requires the OSU population: AgencyNumber = {OSU_AGENCY_NUMBER}.
Unless the user explicitly requests another available year, use Year = {selected_year}.
Never query any table other than pcards. Return exactly one SQLite SELECT statement (a WITH
clause is allowed).
Do not use semicolons, PRAGMA, write operations, schema operations, ATTACH, or comments.

Notes about the data:
- Each row is one P-card transaction. FullName identifies the cardholder (e.g. 'Employee 51914657').
- Amount is in US dollars. Negative amounts are credits or returns.
- Year and Month are integers taken from the transaction date.
- TransactionDate and PostedDate are TEXT in the format 'M/D/YYYY 0:00:00'. To sort or compare
  dates, build an ISO date with: {ISO_DATE_SQL}
- MCC is the merchant category as text (e.g. 'EATING PLACES AND RESTAURANTS').
- Text comparisons with LIKE are case-insensitive. Round money to 2 decimals.
- Add a LIMIT of at most 1000 rows unless the user asks for a full list.

Return JSON with exactly these string keys:
- sql: the SQLite query
- explanation: a concise explanation of the period, filters, grouping, and sorting

Auditor question: {question}"""


class GeminiBusyError(Exception):
    """Gemini is temporarily overloaded or rate-limited after all retries."""


RETRYABLE_STATUS = (429, 500, 503, 504)
RETRY_WAITS_SECONDS = (2, 5)  # waits between attempts: 3 attempts in total


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in RETRYABLE_STATUS:
        return True
    text = str(exc).upper()
    return any(marker in text for marker in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


def ask_gemini(prompt: str, api_key: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = None
    for attempt in range(len(RETRY_WAITS_SECONDS) + 1):
        try:
            response = client.models.generate_content(
                model=get_setting("GEMINI_MODEL", DEFAULT_MODEL),
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=1024,
                ),
            )
            break
        except Exception as exc:  # retry only temporary overload / rate-limit errors
            if not _is_retryable(exc):
                raise
            if attempt == len(RETRY_WAITS_SECONDS):
                raise GeminiBusyError(str(exc)) from exc
            time.sleep(RETRY_WAITS_SECONDS[attempt])
    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    data = json.loads(response.text)
    if not isinstance(data, dict) or "sql" not in data:
        raise ValueError("Gemini did not return the expected JSON keys.")
    return {"sql": str(data["sql"]), "explanation": str(data.get("explanation", ""))}


FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|"
    r"ANALYZE|TRIGGER|TRANSACTION|BEGIN|COMMIT|ROLLBACK|SAVEPOINT|LOAD_EXTENSION)\b",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> str:
    """Check the generated SQL before it is run. Returns the cleaned SQL or raises ValueError.
    The read-only connection and the authorizer give a second layer of protection."""
    cleaned = sql.strip().rstrip(";").strip()
    without_strings = re.sub(r"'(?:[^']|'')*'", "''", cleaned)  # ignore text inside quotes
    if not cleaned:
        raise ValueError("The query is empty.")
    if ";" in without_strings:
        raise ValueError("Only one statement is allowed (no semicolons).")
    if "--" in without_strings or "/*" in without_strings:
        raise ValueError("Comments are not allowed in the query.")
    if not re.match(r"^\s*(SELECT|WITH)\b", without_strings, re.IGNORECASE):
        raise ValueError("Only SELECT queries (optionally starting with WITH) are allowed.")
    match = FORBIDDEN_KEYWORDS.search(without_strings)
    if match:
        raise ValueError(f"The keyword {match.group(1).upper()} is not allowed.")
    return cleaned


def show_results(df: pd.DataFrame, truncated: bool, file_name: str, key: str) -> None:
    if df.empty:
        st.info("The query returned no rows.")
        return
    note = f" (showing the first {MAX_DISPLAY_ROWS:,})" if truncated else ""
    st.caption(f"{len(df):,} rows{note}")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        "Download results as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )


def tab_ask(years: list[int]) -> None:
    st.subheader("Ask the database")
    st.markdown(
        "Ask a question about OSU P-card transactions in plain English. The app sends your "
        "question and the table structure to Gemini, which writes one SQL query. The app checks "
        "that the query is read-only, runs it on the OSU data and shows the result."
    )
    with st.expander("Example questions"):
        st.markdown(
            "- Which 10 cardholders spent the most in total?\n"
            "- Show all transactions above $5,000, largest first.\n"
            "- Total spending per month, January first.\n"
            "- Which vendors in the MCC 'EATING PLACES AND RESTAURANTS' received the most money?\n"
            "- List cardholders with more than 5 transactions at liquor stores."
        )

    api_key = gemini_secret()
    if not api_key:
        st.warning(
            "No Gemini API key found. Add GEMINI_API_KEY to the app's secrets "
            "(Streamlit Cloud: Settings, Secrets) or to a local .env file."
        )

    with st.form("ask_form"):
        col_year, col_q = st.columns([1, 4])
        year = col_year.selectbox("Year", years, index=0, key="ask_year")
        question = col_q.text_input("Question", placeholder="e.g. Which cardholders spent more than $50,000?")
        submitted = st.form_submit_button("Ask", type="primary", disabled=not api_key)

    if not submitted:
        return
    if not question.strip():
        st.error("Please enter a question.")
        return

    with st.spinner("Asking Gemini to write the query..."):
        try:
            answer = ask_gemini(build_prompt(question.strip(), year), api_key)
        except GeminiBusyError:
            st.warning(
                "Gemini is busy right now (high demand or rate limit) and did not answer after "
                "three attempts. Please wait a minute and click Ask again. "
                "The Prohibited purchases dashboard does not use Gemini and works in the meantime."
            )
            return
        except Exception as exc:  # network, invalid key, JSON errors
            st.error(f"Gemini could not answer the question: {exc}")
            return

    st.markdown("**Explanation**")
    st.write(answer["explanation"] or "No explanation returned.")
    st.markdown("**SQL**")
    st.code(answer["sql"], language="sql")

    try:
        sql = validate_sql(answer["sql"])
    except ValueError as exc:
        st.error(f"The generated query was rejected: {exc}")
        return

    try:
        df, truncated = run_query(sql)
    except sqlite3.DatabaseError as exc:
        st.error(f"The query could not be run: {exc}")
        return

    st.markdown("**Results**")
    show_results(df, truncated, f"pcard_question_{year}.csv", key="ask_download")


# ---------------------------------------------------------------------------
# Tab 2: prohibited purchases dashboard
# ---------------------------------------------------------------------------
PROHIBITED = pd.DataFrame(
    [
        ("Alcohol", "liquor, wine, beer, brew, pub", "liquor, bottle shop, brewery, tavern"),
        ("Cash, cash advances, ATM", "cash, ATM, advance", "ATM, cash, western union, moneygram"),
        ("Decorations", "decoration, decor, balloon", "party, decor, balloon"),
        ("Donations and sponsorships", "donation, sponsor", "giving, foundation, network for good, BLK"),
        ("Gasoline", "fuel, gasoline, unleaded, diesel", "oil, fuel, shell, chevron, conoco, quiktrip"),
        ("Gifts, gift cards, gift certificates", "gift card, gift cert", "gift, florist, flowers, hallmark"),
        ("Insurance", "insurance", "insurance, state farm"),
        ("Late fees", "late fee, late charge, finance charge", ""),
        ("Mail and postage", "postage, stamps", "USPS, post office, stamps.com"),
        ("Moving expenses", "moving, movers", "moving, U-Haul, storage"),
        ("Personal purchases", "personal", "walmart, target, best buy, dental, eyecare"),
        ("Personal memberships and dues", "membership, dues", "alumni, association, club"),
        ("Salaries, wages and benefits", "salary, wage, payroll", "payroll, ADP"),
        ("Service and incentive awards", "award, plaque, trophy", "awards, trophy, engraving"),
    ],
    columns=["Prohibited category", "Description keywords to try", "Vendor keywords to try"],
)

SEARCH_COLUMNS = (
    "ID, Amount, FullName, Description, Vendor, TransactionDate, PostedDate, MCC"
)


def escape_like(keyword: str) -> str:
    """Escape LIKE wildcards so the auditor's keyword is matched literally."""
    return keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def keyword_search(field: str, year: int, keyword: str, limit: int):
    """Fixed SQL template. {field} is chosen by the app (Description or Vendor), never by the
    user; the year, keyword and limit are bound as parameters."""
    if field not in ("Description", "Vendor"):
        raise ValueError("Invalid search field.")
    pattern = f"%{escape_like(keyword)}%"
    where = f"""WHERE AgencyNumber = ?
  AND Year = ?
  AND {field} IS NOT NULL
  AND {field} LIKE ? ESCAPE '\\'"""
    params = (OSU_AGENCY_NUMBER, year, pattern)
    detail_sql = f"""SELECT {SEARCH_COLUMNS}
FROM pcards
{where}
ORDER BY {ISO_DATE_SQL} DESC, ID DESC
LIMIT ?"""
    summary_sql = f"""SELECT COUNT(*) AS n, ROUND(SUM(Amount), 2) AS total,
       COUNT(DISTINCT FullName) AS cardholders, COUNT(DISTINCT Vendor) AS vendors
FROM pcards
{where}"""
    by_holder_sql = f"""SELECT FullName, COUNT(*) AS Transactions, ROUND(SUM(Amount), 2) AS TotalAmount
FROM pcards
{where}
GROUP BY FullName
ORDER BY TotalAmount DESC
LIMIT 10"""
    detail, truncated = run_query(detail_sql, params + (limit,))
    summary, _ = run_query(summary_sql, params)
    by_holder, _ = run_query(by_holder_sql, params)
    return detail, truncated, summary.iloc[0], by_holder


def search_section(field: str, year: int, title: str, help_text: str, placeholder: str) -> None:
    st.markdown(f"### {title}")
    st.caption(help_text)
    with st.form(f"{field}_form"):
        col_kw, col_limit = st.columns([3, 1])
        keyword = col_kw.text_input("Keyword", placeholder=placeholder, key=f"{field}_kw")
        limit = col_limit.number_input(
            "Max rows", min_value=10, max_value=MAX_DISPLAY_ROWS, value=500, step=50, key=f"{field}_limit"
        )
        submitted = st.form_submit_button(f"Search {field.lower()}", type="primary")
    if not submitted:
        return
    if len(keyword.strip()) < 2:
        st.error("Please enter a keyword of at least two characters.")
        return
    try:
        detail, truncated, summary, by_holder = keyword_search(field, year, keyword.strip(), int(limit))
    except sqlite3.DatabaseError as exc:
        st.error(f"The search could not be run: {exc}")
        return
    if summary["n"] == 0:
        st.info(f"No {year} transactions with '{keyword.strip()}' in the {field} field.")
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions", f"{int(summary['n']):,}")
    m2.metric("Total amount", f"${summary['total']:,.2f}")
    m3.metric("Cardholders", f"{int(summary['cardholders']):,}")
    m4.metric("Vendors", f"{int(summary['vendors']):,}")
    with st.expander("Top cardholders in these results"):
        st.dataframe(by_holder, width="stretch", hide_index=True)
    shown = f"Showing the {len(detail):,} most recent" if truncated or len(detail) < summary["n"] else "Showing all"
    st.caption(f"{shown} of {int(summary['n']):,} matching transactions, newest first.")
    st.dataframe(detail, width="stretch", hide_index=True)
    st.download_button(
        "Download results as CSV",
        detail.to_csv(index=False).encode("utf-8"),
        file_name=f"pcard_{field.lower()}_{keyword.strip().replace(' ', '_')}_{year}.csv",
        mime="text/csv",
        key=f"{field}_download",
    )


def tab_dashboard(years: list[int]) -> None:
    st.subheader("Prohibited purchases dashboard")
    with st.expander("How to use this dashboard", expanded=True):
        st.markdown(
            "1. **Select the year** you want to examine.\n"
            "2. Use **Description search** to look for a keyword in the transaction description "
            "entered by the cardholder (for example *gift card* or *postage*).\n"
            "3. Use **Vendor search** to look for a keyword in the merchant name "
            "(for example *USPS* for the post office or *liquor* for liquor stores).\n"
            "4. Matching is case-insensitive and finds the keyword anywhere in the field. "
            "Short keywords can match unrelated words (\"wine\" also matches \"swine\"), so check the "
            "Vendor and MCC columns before drawing conclusions.\n"
            "5. Review the results, then **download them as CSV** for follow-up. "
            "A match is a potential deviation, not proof of a violation: request the receipt and "
            "business purpose before concluding."
        )
    with st.expander("Prohibited purchase categories and suggested keywords"):
        st.dataframe(PROHIBITED, width="stretch", hide_index=True)

    year = st.selectbox("Year to search", years, index=0, key="dash_year")
    st.divider()
    search_section(
        "Description",
        year,
        "1. Description search",
        "Searches the Description field (what was bought, as entered by the cardholder).",
        "e.g. gift card",
    )
    st.divider()
    search_section(
        "Vendor",
        year,
        "2. Vendor search",
        "Searches the Vendor field (the merchant's name).",
        "e.g. USPS",
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("OSU P-card Audit")
    st.caption(
        "Internal audit tool for Oklahoma State University purchasing-card transactions. "
        "All queries are read-only and limited to OSU (agency 1000)."
    )
    if not DB_PATH.exists():
        st.error("The database file pcards.db was not found next to app.py.")
        st.stop()
    years = available_years()
    tab1, tab2 = st.tabs(["Ask the database", "Prohibited purchases dashboard"])
    with tab1:
        tab_ask(years)
    with tab2:
        tab_dashboard(years)


main()
