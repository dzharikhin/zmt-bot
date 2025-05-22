# zmt-bot
app to recommend track based on user audio profile

## Build image
```shell
VER=$(poetry version --short) docker buildx bake --progress=plain tg-zmt-bot
```

## Run
```shell
export API_HASH=
export API_ID=
export BOT_TOKEN=
export OWNER_USER
docker run --rm -d --name "tg-zmt-bot" -v "./data:/app/data" --env API_HASH --env API_ID --env BOT_TOKEN --env OWNER_USER_ID "tg-zmt-bot:$(poetry version --short)"
```
