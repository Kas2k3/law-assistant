import os

from dotenv import load_dotenv
from peewee import MySQLDatabase

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456789")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "law")

db = MySQLDatabase(
    database=MYSQL_DATABASE,
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
)

