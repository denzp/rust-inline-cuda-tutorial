use std::process::Command;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let builder = XargoBuilder::new("../device", "chapter_1_kernel");

    #[cfg(not(rls))]
    match builder.build() {
        Ok(output) => output.announce_ptx_path("KERNEL_PTX_PATH"),

        Err(output) => {
            println!("cargo:warning=PTX build failed");

            for line in output.split('\n') {
                println!("cargo:warning=Xargo: {}", line);
            }
        }
    }
}

struct XargoBuilder {
    crate_path: PathBuf,
    crate_name: String,
}

#[derive(Debug)]
struct XargoBuilderResult {
    output_path: PathBuf,
    crate_name: String,
}

impl XargoBuilder {
    pub fn new(device_crate_path: &str, device_crate_name: &str) -> XargoBuilder {
        XargoBuilder {
            crate_name: String::from(device_crate_name),
            crate_path: PathBuf::from(env::current_dir().unwrap()).join(device_crate_path),
        }
    }

    pub fn build(self) -> Result<XargoBuilderResult, String> {
        let output_path = env::temp_dir().join(format!("rust-ptx/{}", self.crate_name));

        let xargo_result = Command::new("xargo")
            .current_dir(self.crate_path.as_path())
            .args(&["build", "--release", "--target", "nvptx64-nvidia-cuda"])
            .env("CARGO_TARGET_DIR", output_path.as_path())
            .env("RUST_TARGET_PATH", self.crate_path.as_path())
            .output();

        match xargo_result {
            Ok(output) => {
                if output.status.success() {
                    Ok(XargoBuilderResult::new(
                        &output_path.as_path(),
                        &self.crate_name,
                    ))
                } else {
                    let mut lines = String::from_utf8(output.stdout).unwrap();

                    lines += "\n";
                    lines += &String::from_utf8(output.stderr).unwrap();

                    Err(lines)
                }
            }

            Err(reason) => Err(format!("xargo failed because {}", reason)),
        }
    }
}

impl XargoBuilderResult {
    pub fn new(output_path: &Path, crate_name: &str) -> XargoBuilderResult {
        XargoBuilderResult {
            output_path: PathBuf::from(output_path),
            crate_name: String::from(crate_name),
        }
    }

    pub fn announce_ptx_path(&self, env_name: &str) {
        let mut ptx_path = self.output_path.clone();

        ptx_path.push("nvptx64-nvidia-cuda/release");
        ptx_path.push(&self.crate_name);
        ptx_path.set_extension("ptx");

        println!(
            "cargo:rustc-env={}={}",
            env_name,
            ptx_path.to_str().unwrap()
        );
    }
}
