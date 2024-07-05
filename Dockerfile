FROM python

WORKDIR  /chatbot

COPY . .

RUN pip install -r requirements.txt

EXPOSE 3000

CMD [ "python", "main.py" ]