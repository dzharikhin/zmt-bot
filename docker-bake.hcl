variable "VER" {}
variable "SKIP_LLVM_TESTS" {
  default = "1"
}
target "tg-zmt-bot" {
    context = "."
    dockerfile = "Dockerfile"
    args = {
        SKIP_LLVM_TESTS = "${SKIP_LLVM_TESTS}"
    }
    tags = [ "tg-zmt-bot:${VER}" ]
    output = [{ type = "docker" }]
}