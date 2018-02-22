extern crate ptx_builder;

use std::process::exit;
use ptx_builder::prelude::*;

#[cfg(rls)]
fn main() {}

#[cfg(not(rls))]
fn main() {
    if let Err(error) = build() {
        eprintln!("{}", BuildReporter::report(error));
        exit(1);
    }
}

fn build() -> Result<()> {
    let output = Builder::new("../device")?.build()?;

    // Provide the PTX Assembly location via env variable
    println!(
        "cargo:rustc-env=KERNEL_PTX_PATH={}",
        output.get_assembly_path().to_str().unwrap()
    );

    // Observe changes in kernel sources
    for path in output.source_files()? {
        println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
    }

    Ok(())
}
