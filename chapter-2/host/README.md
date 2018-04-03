# Chapter 2
> Merging CPU and GPU code into one crate.

## Introduction
After first success with CUDA, it's time to add safety for our code.
Unfortunately, we can't just blindly merge and forget both crates into one, because a **most** of the code needs to be compiled for **only single target**, while only a small amount of the code needs to be shared.

## TODO: conditional dependencies

## TODO: conditional crate attributes

## TODO: conditional crate attributes

## TODO: kernel arguments safety
