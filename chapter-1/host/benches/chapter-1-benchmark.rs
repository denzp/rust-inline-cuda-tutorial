#[macro_use]
extern crate criterion;
use criterion::Criterion;

extern crate chapter_1;
use chapter_1::image::Image;

fn cuda_bench(criterion: &mut Criterion) {
    use chapter_1::filter::bilateral_cuda as filter;

    let input_512 = Image::open("../../fixtures/input-512.png").unwrap();
    let input_1024 = Image::open("../../fixtures/input-1024.png").unwrap();
    let input_2048 = Image::open("../../fixtures/input-2048.png").unwrap();
    let input_4096 = Image::open("../../fixtures/input-4096.png").unwrap();

    criterion
        .sample_size(20)
        .without_plots()
        .bench_function("cuda-512", |b| {
            b.iter(|| filter(&input_512, 5, 3.5, 3.0).unwrap())
        });

    criterion
        .sample_size(20)
        .without_plots()
        .bench_function("cuda-1024", |b| {
            b.iter(|| filter(&input_1024, 5, 3.5, 3.0).unwrap())
        });

    criterion
        .sample_size(20)
        .without_plots()
        .bench_function("cuda-2048", |b| {
            b.iter(|| filter(&input_2048, 5, 3.5, 3.0).unwrap())
        });

    criterion
        .sample_size(20)
        .without_plots()
        .bench_function("cuda-4096", |b| {
            b.iter(|| filter(&input_4096, 5, 3.5, 3.0).unwrap())
        });
}

criterion_group!(benches, cuda_bench);
criterion_main!(benches);
