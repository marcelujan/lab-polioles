"""Crea o actualiza un usuario de Firebase Auth y lo habilita para usar la app.

Uso:
python admin_create_user.py \
  --service-account ./service-account.json \
  --email usuario@dominio.com \
  --password 'clave-segura' \
  --role editor
"""

import argparse
from typing import Optional

import firebase_admin
from firebase_admin import auth, credentials



def init_firebase(service_account_path: str) -> None:
    if firebase_admin._apps:
        return
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)



def get_or_create_user(email: str, password: str):
    try:
        user = auth.get_user_by_email(email)
        user = auth.update_user(user.uid, password=password, disabled=False)
        action = "updated"
    except auth.UserNotFoundError:
        user = auth.create_user(
            email=email,
            password=password,
            email_verified=True,
            disabled=False,
        )
        action = "created"
    return user, action



def set_access(uid: str, role: str, can_use_app: bool = True) -> None:
    auth.set_custom_user_claims(
        uid,
        {
            "can_use_app": can_use_app,
            "role": role,
        },
    )



def disable_access_by_email(email: str) -> None:
    user = auth.get_user_by_email(email)
    auth.set_custom_user_claims(
        user.uid,
        {
            "can_use_app": False,
            "role": "auditor",
        },
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-account", required=True, help="Ruta al JSON de service account")
    parser.add_argument("--email", required=True, help="Email del usuario")
    parser.add_argument("--password", required=True, help="Contraseña inicial")
    parser.add_argument("--role", default="editor", help="Rol a asignar")
    parser.add_argument(
        "--disable-access",
        action="store_true",
        help="Deja el usuario autenticable pero sin permiso de uso en la app",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    init_firebase(args.service_account)

    user, action = get_or_create_user(args.email, args.password)
    if args.disable_access:
        disable_access_by_email(args.email)
        print(f"User {action}: {user.uid} | access disabled | email={args.email}")
        return

    set_access(user.uid, role=args.role, can_use_app=True)
    print(f"User {action}: {user.uid} | role={args.role} | can_use_app=True | email={args.email}")


if __name__ == "__main__":
    main()
