use cuda::driver;
use cuda::driver::{Any, Block, Context, Device, Function, Grid, Module};
use std::ffi::CString;
use std::marker::PhantomData;
use std::ops::Deref;

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

    pub fn kernel<F: KernelPlaceholder>(&'static self) -> Result<Kernel<F>, driver::Error> {
        Kernel::<F>::new(self)
    }
}

impl Deref for StaticModule {
    type Target = Module<'static>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait KernelPlaceholder {
    type Args;

    fn get_name() -> &'static str;
}

pub struct Kernel<F: KernelPlaceholder> {
    handle: Function<'static, 'static>,
    signature: PhantomData<F>,
}

impl<F: KernelPlaceholder> Kernel<F> {
    fn new(module: &'static StaticModule) -> Result<Self, driver::Error> {
        let kernel_name = CString::new(F::get_name()).expect("Unable to create kernel name string");

        Ok(Kernel {
            handle: module.function(&kernel_name)?,
            signature: PhantomData::default(),
        })
    }
}

pub mod prelude {
    pub use super::ModuleKernelWithArity5;
}

pub trait ModuleKernelWithArity5<I1, I2, I3, I4, I5> {
    fn execute(
        &self,
        grid: Grid,
        block: Block,
        i1: I1,
        i2: I2,
        i3: I3,
        i4: I4,
        i5: I5,
    ) -> Result<(), driver::Error>;
}

impl<F, I1, I2, I3, I4, I5> ModuleKernelWithArity5<I1, I2, I3, I4, I5> for Kernel<F>
where
    F: KernelPlaceholder<Args = (I1, I2, I3, I4, I5)>,
{
    fn execute(
        &self,
        grid: Grid,
        block: Block,
        i1: I1,
        i2: I2,
        i3: I3,
        i4: I4,
        i5: I5,
    ) -> Result<(), driver::Error> {
        self.handle.launch(
            &[Any(&i1), Any(&i2), Any(&i3), Any(&i4), Any(&i5)],
            grid,
            block,
        )
    }
}
