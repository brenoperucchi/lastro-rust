fn main() {
    let cuda = std::env::var("CUDA_TOOLKIT_PATH").unwrap_or_else(|_| "/opt/cuda".into());
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
}
