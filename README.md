# OSU P-card audit app

Streamlit app for internal auditors examining Oklahoma State University P-card transactions (`pcards.db`, SQLite).

**Live app:** https://pcard-audit-app.streamlit.app

**Tab 1: Ask the database.** Natural-language questions are translated into a single read-only SQLite query by Gemini on the backend. The app validates the SQL, runs it against a read-only, OSU-only view of the data and shows the explanation, the query, the results and a CSV download.

**Tab 2: Prohibited purchases dashboard.** Instructions, a year selector and two separate keyword searches (Description and Vendor) using a fixed, parameterised SQL template (no LLM call). Results show summary figures, the top cardholders and all transaction details with a CSV download.

## Security

- The Gemini API key is never stored in the code or in this repository. It is read from Streamlit secrets (`.streamlit/secrets.toml` locally, the Secrets editor on Streamlit Community Cloud) or from a local `.env` file. Both are listed in `.gitignore`.
- Temporary Gemini overload or rate-limit errors (429/503) are retried automatically up to three times.
- Generated SQL is checked (single SELECT/WITH statement, no semicolons, comments, PRAGMA, ATTACH or write operations).
- The database is opened read-only, and an SQLite authorizer only allows reading a view restricted to OSU (AgencyNumber = 1000).

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then insert your own GEMINI_API_KEY in .env
streamlit run app.py
```

## Files

- `app.py`: Streamlit app
- `requirements.txt`: Python packages
- `pcards.db`: P-card transaction database (SQLite)
- `.env.example`: template for the local API key file

Data: Oklahoma Purchase Card (P-card) data, made available under the Open Database License. Some transactions were modified for teaching purposes, so the data should not be used to infer intent by anyone listed.
