#[macro_use]
extern crate criterion;
use criterion::Criterion;

extern crate chapter_0;
use chapter_0::image::Image;

fn parallel_bench(criterion: &mut Criterion) {
    use chapter_0::filter::bilateral_parallel as filter;

    let input_512 = Image::open("../../fixtures/input-512.png").unwrap();
    let input_1024 = Image::open("../../fixtures/input-1024.png").unwrap();
    let input_2048 = Image::open("../../fixtures/input-2048.png").unwrap();

    criterion
        .sample_size(20)
        .without_plots()
        .bench_function("parallel-512", |b| {
            b.iter(|| filter(&input_512, 5, 3.5, 3.0))
        });

    criterion
        .sample_size(20)
        .without_plots()
        .bench_function("parallel-1024", |b| {
            b.iter(|| filter(&input_1024, 5, 3.5, 3.0))
        });

    criterion
        .sample_size(20)
        .without_plots()
        .bench_function("parallel-2048", |b| {
            b.iter(|| filter(&input_2048, 5, 3.5, 3.0))
        });
}

fn sequential_bench(criterion: &mut Criterion) {
    use chapter_0::filter::bilateral_sequential as filter;

    let input_512 = Image::open("../../fixtures/input-512.png").unwrap();
    let input_1024 = Image::open("../../fixtures/input-1024.png").unwrap();
    let input_2048 = Image::open("../../fixtures/input-2048.png").unwrap();

    criterion
        .sample_size(20)
        .without_plots()
        .bench_function("sequential-512", |b| {
            b.iter(|| filter(&input_512, 5, 3.5, 3.0))
        });

    criterion
        .sample_size(20)
        .without_plots()
        .bench_function("sequential-1024", |b| {
            b.iter(|| filter(&input_1024, 5, 3.5, 3.0))
        });

    criterion
        .sample_size(20)
        .without_plots()
        .bench_function("sequential-2048", |b| {
            b.iter(|| filter(&input_2048, 5, 3.5, 3.0))
        });
}

criterion_group!(benches, parallel_bench, sequential_bench);
criterion_main!(benches);
