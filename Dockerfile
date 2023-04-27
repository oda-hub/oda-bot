FROM python:3.9

COPY . /source

RUN pip install -r /source/requirements.txt && pip install /source && rm -r /source

RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" &&\
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && rm ./kubectl

ENTRYPOINT ["odabot", "--settings", "/settings.toml", "update-workflows", "--loop", "300"]