#!/usr/bin/env python
"""
Create a user in the RiskModel database.

Usage:
    python scripts/create_user.py <username> <password> <role>

Arguments:
    username  — unique username
    password  — plaintext password (will be hashed before storage)
    role      — 'admin' or 'analyst'
"""

import argparse
import os
import sys

# Ensure project root is on the path so imports work when run from any cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

load_dotenv()

from config import Config
from database import create_tables
from auth import create_user


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a user in the RiskModel database.")
    parser.add_argument("username", help="Unique username")
    parser.add_argument("password", help="Plaintext password (will be hashed)")
    parser.add_argument("role", choices=["admin", "analyst"], help="User role")
    args = parser.parse_args()

    db_path = Config.DATABASE_URI
    create_tables(db_path)
    create_user(db_path, args.username, args.password, args.role)
    print(f"User '{args.username}' created with role '{args.role}'.")


if __name__ == "__main__":
    main()
