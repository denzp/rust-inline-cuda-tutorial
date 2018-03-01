use std::ffi::CString;
use std::ops::Deref;
use cuda::driver;
use cuda::driver::{Context, Device, Function, Module};

pub struct StaticContext(Context);
pub struct StaticModule(Module<'static>);

lazy_static! {
    pub static ref CUDA_CTX: StaticContext = StaticContext::new();
    pub static ref CUDA_MODULE: StaticModule = StaticModule::new(&CUDA_CTX);
}

unsafe impl Send for StaticContext {}
unsafe impl Sync for StaticContext {}

unsafe impl Send for StaticModule {}
unsafe impl Sync for StaticModule {}

impl StaticContext {
    pub fn new() -> Self {
        driver::initialize().expect("Unable to initialize CUDA");

        StaticContext(
            Device(0)
                .expect("Unable to get CUDA device 0")
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
        let ptx_bytecode = CString::new(include_str!(env!("KERNEL_PTX_PATH")));

        StaticModule(
            context
                .load_module(&ptx_bytecode.expect("Unable to create sources"))
                .expect("Unable to create module"),
        )
    }

    pub fn kernel<'a>(&'a self, name: &str) -> Result<Function<'static, 'a>, driver::Error> {
        let kernel_name = CString::new(name).expect("Unable to create kernel name string");

        self.function(&kernel_name)
    }
}

impl Deref for StaticModule {
    type Target = Module<'static>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
