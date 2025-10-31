# zmt-bot
app to recommend track based on user audio profile
to see available commands - call `/start`

## Data
Prepared data in releases on github

## Build
> requires installed [poetry](https://python-poetry.org/)

```shell
VER=$(poetry version --short) docker buildx bake --progress=plain tg-zmt-bot
```

## Run
```shell
API_HASH= API_ID= BOT_TOKEN= OWNER_USER_ID= docker run -d --restart unless-stopped --name "tg-zmt-bot" -v "./data:/app/data" --env API_HASH --env API_ID --env BOT_TOKEN --env OWNER_USER_ID --memory=2G  --cpus=3 "tg-zmt-bot:$(poetry version --short)"
```

## Export
```shell
docker save "tg-zmt-bot:$(poetry version --short)" > "tg-zmt-bot_$(poetry version --short)".tar
```
# Essentia

## Build
`docker buildx bake --progress=plain essentia-builder`
> to drop cache for selected stages: `--set essentia-builder.no-cache-filter=builder`

## Add as dependency
`poetry add --editable essentia-wheels/essentia-2.1b6.dev0-cp312-cp312-manylinux_2_35_x86_64.whl`
> look at wheels size - the biggest is what you need  
> `--editable` - for local development