FROM python:3.12-slim

WORKDIR /app

RUN pip install most-queue pytest

COPY *.py ./

COPY units/*.py ./units/

COPY default_params.yaml ./tests/

CMD ["pytest", "."]
