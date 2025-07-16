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
variable "TARGET_ESSENTIA_COMMIT" {
    default = "bd793a3e07caf1c8c44a8425d6f550bfbc96ede9"
}
target "essentia-builder" {
    context = "."
    dockerfile = "essentia/essentia-tensorflow.Dockerfile"
    args = {
        TARGET_ESSENTIA_COMMIT = "${TARGET_ESSENTIA_COMMIT}"
    }
    tags = [ "essentia-builder:${TARGET_ESSENTIA_COMMIT}" ]
    output = [{ type = "local", dest = "." }]
}