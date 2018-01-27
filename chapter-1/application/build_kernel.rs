use std::process::Command;
use std::env;
use std::path::PathBuf;

fn main() {
    #[cfg(not(rls))]
    match build_ptx() {
        Ok(_) => {}

        Err(reason) => {
            println!("cargo:warning=PTX build failed: {}", reason);
        }
    }
}

fn build_ptx() -> Result<(), String> {
    let mut kernel_root_path = PathBuf::from(env::current_dir().unwrap());
    kernel_root_path.pop();
    kernel_root_path.push("kernel");

    let xargo_result = Command::new("xargo")
        .current_dir(kernel_root_path.as_path())
        .env("RUST_TARGET_PATH", kernel_root_path.as_path())
        .args(&["build", "--release", "--target", "nvptx64-nvidia-cuda"])
        .output();

    let mut kernel_output_path = PathBuf::from(env::current_dir().unwrap());
    kernel_output_path.pop();
    kernel_output_path.pop();
    kernel_output_path.push("target");
    kernel_output_path.push("nvptx64-nvidia-cuda");
    kernel_output_path.push("release");
    kernel_output_path.push("chapter_1_kernel.ptx");

    println!(
        "cargo:rustc-env=KERNEL_PTX_PATH={}",
        kernel_output_path.to_str().unwrap()
    );

    match xargo_result {
        Ok(output) => {
            if output.status.success() {
                Ok(())
            } else {
                report_ptx_errors(&String::from_utf8(output.stderr).unwrap());
                Err(format!("xargo exited with code {}", output.status))
            }
        }

        Err(reason) => Err(format!("xargo failed because {}", reason)),
    }
}

fn report_ptx_errors(stderr: &str) {
    for line in stderr.split('\n') {
        println!("cargo:warning=Xargo: {}", line);
    }
}
