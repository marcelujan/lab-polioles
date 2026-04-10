import argparse
import json
from pathlib import Path

import firebase_admin
from firebase_admin import auth, credentials


def init_app(service_account_path: str):
    if firebase_admin._apps:
        return firebase_admin.get_app()
    cred = credentials.Certificate(service_account_path)
    return firebase_admin.initialize_app(cred)


def create_or_update_user(email: str, password: str, role: str, can_use_app: bool):
    try:
        user = auth.get_user_by_email(email)
        auth.update_user(user.uid, password=password)
        uid = user.uid
    except auth.UserNotFoundError:
        user = auth.create_user(email=email, password=password)
        uid = user.uid

    auth.set_custom_user_claims(
        uid,
        {
            "can_use_app": bool(can_use_app),
            "role": role,
        },
    )
    return uid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--role", default="editor")
    parser.add_argument("--disable-app", action="store_true")
    args = parser.parse_args()

    init_app(args.service_account)
    uid = create_or_update_user(
        email=args.email,
        password=args.password,
        role=args.role,
        can_use_app=not args.disable_app,
    )
    print(json.dumps({"uid": uid, "email": args.email, "role": args.role, "can_use_app": not args.disable_app}, ensure_ascii=False))


if __name__ == "__main__":
    main()
