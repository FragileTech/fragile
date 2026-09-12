//! Diagnostic timings, not a machine-dependent pass/fail test.
use algorithmic_gas::{
    donor::{CompanionRequest, CompanionSampler, DonorModule, DonorPool, SamplingLaw},
    geometry::Kernel,
    random::Stream,
    *,
};
use futures_lite::future::block_on;
fn main() -> Result<()> {
    block_on(async {
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F32).await?;
        for law in [SamplingLaw::Independent, SamplingLaw::FisherYates] {
            for n in [2000, 4000, 8000, 16000] {
                let p = Population::new(ObservationBatch::positions(TensorBatch::vectors(
                    n,
                    1,
                    vec![0_f32; n],
                )?))?;
                let pool = DonorPool::freeze(&p, 0, &[], 0, false)?;
                let alive = vec![true; n];
                let module = DonorModule {
                    law,
                    kernel: Kernel::Uniform,
                    ..Default::default()
                };
                let start = std::time::Instant::now();
                for seed in 0..30 {
                    let output = module
                        .sample(
                            CompanionRequest {
                                population: &p,
                                pool: &pool,
                                eligible: &alive,
                                seed,
                                step: 1,
                                stream: Stream::Distance,
                            },
                            &mut cx,
                        )
                        .await?;
                    std::hint::black_box(output);
                }
                println!(
                    "{law:?}, N={n}, K=1, 30 draws: {:.3} ms",
                    start.elapsed().as_secs_f64() * 1000.
                );
            }
        }
        Ok(())
    })
}
