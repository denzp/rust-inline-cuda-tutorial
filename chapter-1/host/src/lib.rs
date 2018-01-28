#![deny(warnings)]

#[macro_use]
extern crate lazy_static;

extern crate cuda;
extern crate png;

mod static_cuda;

pub mod image;
pub mod filter;
