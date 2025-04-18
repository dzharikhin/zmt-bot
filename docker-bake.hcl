group "default" {
    targets = [ "deps", "tg-zmt-bot" ]
}
variable "TG_ZMT_VERSION" {
}
variable "QEMU_PAGESIZE" {
  default = "32768"
}
variable "VER" {
  default = "${TG_ZMT_VERSION}-armv7-pagesize${QEMU_PAGESIZE}"
}
target "deps" {
    context = "."
    dockerfile = "dependencies.x86_64.to.armv7.Dockerfile"
    ulimits = [
        "nofile=4096:4096"
    ]
    tags = [ "tg-zmt-bot_dependencies:${VER}" ]
    output = [{ type = "docker" }]
}
variable "SKIP_LLVM_TESTS" {
  default = "1"
}
target "tg-zmt-bot" {
    context = "."
    dockerfile = "Dockerfile"
    contexts = {
      dependencies_arm = "target:deps"
    }
    args = {
        QEMU_PAGESIZE = "${QEMU_PAGESIZE}"
        SKIP_LLVM_TESTS = "${SKIP_LLVM_TESTS}"
    }
    tags = [ "tg-zmt-bot:${VER}" ]
    output = [{ type = "docker" }]
    platforms = [ "linux/arm/v7" ]
}