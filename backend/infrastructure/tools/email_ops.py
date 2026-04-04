import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

def send_email(to_address: str, subject: str, body: str) -> str:
    """
    发送电子邮件给指定联系人。
    :param to_address: 收件人邮箱地址
    :param subject: 邮件主题
    :param body: 邮件正文
    """
    # 动态从配置获取，防止硬编码
    smtp_server = getattr(settings, "EMAIL_SMTP_SERVER", "")
    smtp_port = getattr(settings, "EMAIL_SMTP_PORT", 465)
    sender = getattr(settings, "EMAIL_SENDER", "")
    password = getattr(settings, "EMAIL_PASSWORD", "") # 注意：这里通常是应用授权码，不是登录密码
    
    if not all([smtp_server, sender, password]):
        return "【系统拦截】: 邮件服务未配置。请管理员在 .env 中配置 EMAIL_SENDER 和 EMAIL_PASSWORD。"

    logger.info(f"🛠️ [Tool] 正在发送真实邮件至: {to_address}")
    
    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = to_address
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # 默认使用 SSL 连接
        with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=10) as server:
            server.login(sender, password)
            server.send_message(msg)
            
        return f"✅ 邮件已成功投递至 {to_address}。主题: '{subject}'"
        
    except smtplib.SMTPAuthenticationError:
        return "发送失败：SMTP 认证失败，请检查发件人邮箱账号或授权码是否正确。"
    except Exception as e:
        logger.error(f"邮件发送异常: {e}")
        return f"发送失败：{str(e)}"