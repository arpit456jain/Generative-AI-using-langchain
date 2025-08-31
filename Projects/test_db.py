import psycopg2

# Database connection details
DB_HOST = "localhost"
DB_NAME = "demo"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_PORT = "5432"

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

    # Create a cursor
    cur = conn.cursor()

    # Execute a query (change table name to yours)
    cur.execute("SELECT * FROM student LIMIT 5;")

    # Fetch and print results
    rows = cur.fetchall()
    for row in rows:
        print(row)

    # Close cursor and connection
    cur.close()
    conn.close()

except Exception as e:
    print("❌ Error:", e)
