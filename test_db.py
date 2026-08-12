from banking_agent import get_db_connection


connection = get_db_connection()
cursor = connection.cursor()
cursor.execute("SELECT DATABASE()")
print("Connected database:", cursor.fetchone()[0])
cursor.execute("SELECT COUNT(*) FROM customers")
print("Customers:", cursor.fetchone()[0])
cursor.execute("SELECT COUNT(*) FROM transactions")
print("Transactions:", cursor.fetchone()[0])

from banking_agent import execute_sql

result = execute_sql.invoke(
    """
    SELECT
        c.country,
        c.transaction_year,
        SUM(t.amount_debit) AS total_debit,
        t.currency
    FROM customers c
    JOIN transactions t
        ON c.customer_id = t.customer_id
    GROUP BY
        c.country,
        c.transaction_year,
        t.currency
    ORDER BY
        c.country,
        c.transaction_year;
    """
)

print(result)

cursor.close()
connection.close()