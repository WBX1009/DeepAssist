def send_email(to_address: str, subject: str, body: str) -> str:
    """
    发送电子邮件给指定联系人。
    :param to_address: 收件人邮箱地址
    :param subject: 邮件主题
    :param body: 邮件正文
    """
    # 占位实现，实际项目中接入 SMTP
    return f"✅ 邮件已成功投递至 {to_address}。主题: '{subject}'"