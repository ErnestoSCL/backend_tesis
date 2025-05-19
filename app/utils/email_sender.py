from fastapi_mail import FastMail, MessageSchema
from fastapi_mail.config import ConnectionConfig


async def enviar_pdf_por_correo(path: str, destinatario: str, conf: ConnectionConfig):
    message = MessageSchema(
        subject="📎 Informe adjunto",
        recipients=[destinatario],
        body="Adjuntamos el informe PDF solicitado.",
        attachments=[path],
        subtype="plain"
    )
    fm = FastMail(conf)
    await fm.send_message(message)
