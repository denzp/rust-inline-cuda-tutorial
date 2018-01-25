use std::ffi::{CStr, CString};
use std::ops::Deref;
use cuda::driver;
use cuda::driver::{Context, Device, Function, Module};

pub struct StaticContext(Context);
pub struct StaticModule(Module<'static>);
pub struct StaticKernel(Function<'static, 'static>);

lazy_static! {
    pub static ref CUDA_CTX: StaticContext = StaticContext::new();
    pub static ref CUDA_MODULE: StaticModule = StaticModule::new(&CUDA_CTX);
    pub static ref CUDA_KERNEL: StaticKernel = StaticKernel::new(&CUDA_MODULE);
}

unsafe impl Send for StaticContext {}
unsafe impl Sync for StaticContext {}

unsafe impl Send for StaticModule {}
unsafe impl Sync for StaticModule {}

unsafe impl Send for StaticKernel {}
unsafe impl Sync for StaticKernel {}

impl StaticContext {
    pub fn new() -> Self {
        driver::initialize().expect("Unable to initialize CUDA");

        let device = Device(0).expect("Unable to get CUDA device 0");

        StaticContext(
            device
                .create_context()
                .expect("Unable to create CUDA context"),
        )
    }
}

impl Deref for StaticContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl StaticModule {
    pub fn new(context: &'static StaticContext) -> Self {
        let ptx_bytecode = CString::new(include_str!(
            "../../kernel/target/nvptx64-nvidia-cuda/release/deps/chapter_1_kernel.ptx"
        ));

        StaticModule(
            context
                .load_module(&ptx_bytecode.expect("Unable to create sources"))
                .expect("Unable to create module"),
        )
    }
}

impl Deref for StaticModule {
    type Target = Module<'static>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl StaticKernel {
    pub fn new(module: &'static StaticModule) -> Self {
        let kernel_name = CStr::from_bytes_with_nul(b"bilateral_filter\0")
            .expect("Unable to create kernel name string");

        StaticKernel(
            module
                .function(&kernel_name)
                .expect("Unable to find the kernel"),
        )
    }
}

impl Deref for StaticKernel {
    type Target = Function<'static, 'static>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
