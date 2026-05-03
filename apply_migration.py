import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv("ml_engine/.env")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not url or not key:
    print("Missing Supabase credentials")
    exit(1)

supabase = create_client(url, key)

migration_path = "supabase/migrations/20260504000000_rag_schema.sql"

with open(migration_path, "r") as f:
    sql = f.read()

print(f"Applying migration: {migration_path}")

try:
    # Supabase Python client doesn't have a direct 'sql' execution method for raw SQL
    # unless using a postgres connection. However, we can try to use a common RPC 
    # if one exists, but usually we just advise the user to use the SQL Editor.
    # Alternatively, we can try to use 'postgrest' to check if tables exist.
    
    print("SUPABASE SQL EXECUTION LIMITATION:")
    print("The Supabase Python client does not support raw SQL execution for security reasons.")
    print("Please copy the contents of 'supabase/migrations/20260504000000_rag_schema.sql'")
    print("and paste them into the Supabase SQL Editor in your dashboard.")
    
except Exception as e:
    print(f"Error: {e}")
