# zmt-bot
app to recommend track based on user audio profile

## Build
> requires installed [poetry](https://python-poetry.org/)

```shell
VER=$(poetry version --short) docker buildx bake --progress=plain tg-zmt-bot
```

## Run
```shell
API_HASH= API_ID= BOT_TOKEN= OWNER_USER_ID= docker run -d --restart unless-stopped --name "tg-zmt-bot" -v "./data:/app/data" --env API_HASH --env API_ID --env BOT_TOKEN --env OWNER_USER_ID "tg-zmt-bot:$(poetry version --short)"
```

## Export
```shell
docker save "tg-zmt-bot:$(poetry version --short)" > "tg-zmt-bot_$(poetry version --short)".tar
```
