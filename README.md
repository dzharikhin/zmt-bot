# zmt-bot
app to recommend track based on user audio profile

## Build image
> now only x86_64 and armv7 are supported
> if you need to add some specific architecture - add {target}-builder stage

### x86_64
```shell
export TG_ZMT_VERSION=
docker build --build-arg SKIP_LLVM_TESTS=1 -t "tg-zmt-bot:${TG_ZMT_VERSION}" .
```

### armv7(page_size=32k)
https://docs.docker.com/build/building/multi-platform/#qemu
```shell
docker run --privileged --rm tonistiigi/binfmt --install all
docker run --privileged --rm multiarch/qemu-user-static --reset -p yes
docker buildx create --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=-1 --driver-opt env.BUILDKIT_STEP_LOG_MAX_SPEED=-1 --name multiarch --driver docker-container --bootstrap
docker buildx use multiarch
```
> run next on x86_64 host
```shell
export TG_ZMT_VERSION=
docker buildx bake
docker save "tg-zmt-bot:${TG_ZMT_VERSION}-armv7-pagesize32768" > "tg-zmt-bot-${TG_ZMT_VERSION}-armv7-pagesize32768".tar
```

## Run
```shell
export TG_ZMT_VERSION=
export API_HASH=
export API_ID=
export BOT_TOKEN=
export OWNER_USER
docker run --rm -d --name "tg-zmt-bot" -v "./data:/app/data" --env API_HASH --env API_ID --env BOT_TOKEN --env OWNER_USER_ID "tg-zmt-bot:${TG_ZMT_VERSION}"
```
