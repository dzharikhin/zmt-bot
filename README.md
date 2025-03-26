# zmt-bot

app to recommend track based on user audio profile

https://docs.docker.com/build/building/multi-platform/#qemu
```shell
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx create --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=-1 --driver-opt env.BUILDKIT_STEP_LOG_MAX_SPEED=-1 --name multiarch --driver docker-container --bootstrap
docker buildx use multiarch
```

```shell
export TG_ZMT_VERSION=
docker build --build-arg SKIP_LLVM_TESTS=1 -t "tg-zmt-bot:${TG_ZMT_VERSION}" .
docker buildx build --ulimit nofile=4096:4096 --build-arg SKIP_LLVM_TESTS=1 --platform=linux/arm/v7 -t "tg-zmt-bot:${TG_ZMT_VERSION}-arm" .
```
> now only x86_64 and armv7 are supported
> if you need to add some specific architecture - add {target}-builder stage

```shell
export TG_ZMT_VERSION=
export API_HASH=
export API_ID=
export BOT_TOKEN=
export OWNER_USER
docker run --rm -d --name "tg-zmt-bot" -v "./data:/app/data" --env API_HASH --env API_ID --env BOT_TOKEN --env OWNER_USER_ID "tg-zmt-bot:${TG_ZMT_VERSION}"
```
```shell
export TG_ZMT_VERSION=
docker save "tg-zmt-bot:${TG_ZMT_VERSION}-arm" > "tg-zmt-bot_${TG_ZMT_VERSION}-arm".tar
```
