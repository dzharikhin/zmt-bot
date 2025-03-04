# zmt-bot

app to recommend track based on user audio profile

```shell
export TG_ZMT_VERSION=
docker build -t "tg-zmt-bot:${TG_ZMT_VERSION}" .
```
```shell
export TG_ZMT_VERSION=
export API_HASH=
export API_ID=
export BOT_TOKEN=
export OWNER_USER_ID=
docker run --rm -d --name "tg-zmt-bot" -v "./data:/app/data" --env API_HASH --env API_ID --env BOT_TOKEN --env OWNER_USER_ID "tg-zmt-bot:${TG_ZMT_VERSION}"
```
```shell
export TG_ZMT_VERSION=
docker save "tg-zmt-bot:${TG_ZMT_VERSION}" > "tg-zmt-bot_${TG_ZMT_VERSION}".tar
```
