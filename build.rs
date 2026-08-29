//! this build script compiles GEM kernels
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    println!("Building cuda source files for GEM...");
    println!("cargo:rerun-if-changed=csrc");

    #[cfg(feature = "cuda")] {
        let csrc_headers = ucc::import_csrc();
        let mut cl_cuda = ucc::cl_cuda_arch(Some(&[80, 86, 89, 90]), None);
        cl_cuda.ccbin(false);
        // ucc::cl_cuda() hardcodes -std=c++14. From CUDA 13, cooperative_groups.h
        // pulls in libcu++, which hard-errors below C++17. nvcc honours the last
        // -std it is given (emitting an "incompatible redefinition" warning), so
        // appending here overrides ucc without patching the submodule.
        cl_cuda.flag("-std=c++17");
        cl_cuda.flag("-lineinfo");
        cl_cuda.flag("-maxrregcount=128");
        cl_cuda.debug(false).opt_level(3)
            .include(&csrc_headers)
            .files(["csrc/kernel_v1.cu"]);
        cl_cuda.compile("gemcu");
        println!("cargo:rustc-link-lib=static=gemcu");
        println!("cargo:rustc-link-lib=dylib=cudart");
        ucc::bindgen(["csrc/kernel_v1.cu"], "kernel_v1.rs");
        ucc::export_csrc();
        ucc::make_compile_commands(&[&cl_cuda]);
    }
}
