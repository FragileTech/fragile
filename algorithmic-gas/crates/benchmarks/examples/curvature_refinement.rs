use algorithmic_gas::{
    lecture::LectureRequest,
    partv_geometry::MetricPolicy,
    physics::{
        fields::archive_fitness_jet,
        geometry::{FitnessJet, MetricJet, fitness_curvature, metric_curvature},
    },
};
use algorithmic_gas_benchmarks::lecture::LectureSession;
use serde_json::json;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    futures_lite::future::block_on(async {
        let mut s = LectureSession::create(LectureRequest {
            id: "VI-39".into(),
            seed: 7,
            steps: 96,
            parameters: json!({}),
        })
        .await?;
        while !s.done() {
            s.advance(8).await?;
        }
        let e = s.evidence();
        let a = &e.archives[0];
        let ix = a.steps.len() - 1;
        let q = a.steps[ix].before.observations.field("positions")?.row(0)?;
        let x = [q[0], q[1], q[2]];
        let make = |z: &[f64]| {
            let j = FitnessJet::from_jet(&archive_fitness_jet(a, ix, 0, z, 4)?.0)?;
            fitness_curvature(&j, 3., MetricPolicy::Clipped, 1e-8, true)
        };
        let c = make(&x)?;
        println!(
            "analytic {} x {:?} spectrum {:?}",
            c.scalar, x, c.spectrum.eigenvalues
        );
        for h in [
            0.008f64, 0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001,
            0.000005, 0.000002, 0.000001,
        ] {
            let mut first = vec![0.; 27];
            let mut second = vec![0.; 81];
            for i in 0..3 {
                let mut xp = x;
                let mut xm = x;
                xp[i] += h;
                xm[i] -= h;
                let plus = make(&xp)?.spectrum.metric;
                let minus = make(&xm)?.spectrum.metric;
                for k in 0..9 {
                    first[i * 9 + k] = (plus[k] - minus[k]) / (2. * h);
                    second[(i * 3 + i) * 9 + k] =
                        (plus[k] - 2. * c.spectrum.metric[k] + minus[k]) / (h * h);
                }
                for j in 0..i {
                    let mut pp = x;
                    let mut pm = x;
                    let mut mp = x;
                    let mut mm = x;
                    pp[i] += h;
                    pp[j] += h;
                    pm[i] += h;
                    pm[j] -= h;
                    mp[i] -= h;
                    mp[j] += h;
                    mm[i] -= h;
                    mm[j] -= h;
                    let pp = make(&pp)?.spectrum.metric;
                    let pm = make(&pm)?.spectrum.metric;
                    let mp = make(&mp)?.spectrum.metric;
                    let mm = make(&mm)?.spectrum.metric;
                    for k in 0..9 {
                        let v = (pp[k] - pm[k] - mp[k] + mm[k]) / (4. * h * h);
                        second[(i * 3 + j) * 9 + k] = v;
                        second[(j * 3 + i) * 9 + k] = v;
                    }
                }
            }
            let num = metric_curvature(
                &MetricJet {
                    dimension: 3,
                    metric: c.spectrum.metric.clone(),
                    first,
                    second,
                },
                true,
            )?;
            println!(
                "h {h:.8} scalar {:.12} abs {:.12} relative {:.9}",
                num.scalar,
                (num.scalar - c.scalar).abs(),
                (num.scalar - c.scalar).abs() / c.scalar.abs()
            );
        }
        Ok::<(), Box<dyn std::error::Error>>(())
    })
}
