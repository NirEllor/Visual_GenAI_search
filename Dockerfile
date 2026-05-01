FROM ubuntu:latest
LABEL authors="Nir"

ENTRYPOINT ["top", "-b"]