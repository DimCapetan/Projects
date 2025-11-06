import smtplib
from email.message import EmailMessage
from string import Template
from pathlib import Path
import os

os.chdir(Path(__file__).parent)

html = Template(Path('file_path').read_text())
email = EmailMessage()
email['from'] = 'Random Name'
email['to'] = 'randomemail@mail.com'
email['subject'] = 'Game\'s gone'

email.set_content(html.substitute({'name': 'Asteroid_Destroyer'}), 'html')

with smtplib.SMTP(host = 'smtp.mail.com', port = 587) as smtp:
    smtp.ehlo()
    smtp.starttls() # Allows for a secure connection to the server
    smtp.login('username@example_email.com', 'password')
    smtp.send_message(email)
    print('all good boss!')

