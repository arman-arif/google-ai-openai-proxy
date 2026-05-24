FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags='-s -w' -o /out/google-ai-openai-proxy .

FROM alpine:3.20
RUN adduser -D -H proxyuser
USER proxyuser
COPY --from=build /out/google-ai-openai-proxy /usr/local/bin/google-ai-openai-proxy
EXPOSE 8080
ENTRYPOINT ["google-ai-openai-proxy"]
