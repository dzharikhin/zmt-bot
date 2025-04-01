group "default" {
    targets = [ "deps", "tg-zmt-bot" ]
}
variable "TG_ZMT_VERSION" {
  default = "1.0"
}
variable "QEMU_PAGESIZE" {
  default = "32768"
}
variable "VER" {
  default = "${TG_ZMT_VERSION}-armv7-pagesize${QEMU_PAGESIZE}"
}
variable "ALLOCATOR" {
  default = "mimalloc"
}
target "deps" {
    context = "."
    dockerfile = "Dockerfile_dependencies_armv7"
    args = {
        ALLOCATOR = "${ALLOCATOR}"
    }
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