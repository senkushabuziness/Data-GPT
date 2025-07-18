Chainlit Datalayer Setup

We are using Chainlit Datalayer to store persistent chat history. Steps to clone and run: Run this in your terminal:

git clone https://github.com/Chainlit/chainlit-datalayer.git

cd chainlit-datalayer

docker-compose up -d

This sets up Postgres + Prisma + Chainlit Datalayer service locally.

for running the application: starts terminal, install requriements,

1st terminal-> python backend.py
2nd terminal-> chainlit run main.py -w