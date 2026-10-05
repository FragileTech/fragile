//! Independent scalar Dirichlet reference experiment for Chapter 7.
//! Compile with rustc --edition 2024 -O; no changes to the swarm algorithm.
use std::{fs::File,io::{BufWriter,Write}};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let path=std::env::args().nth(1).ok_or("chapter07_diffusion_reference OUTPUT_JSON")?;
    let mut w=BufWriter::new(File::create(path)?);
    writeln!(w,"{{\"chapter\":7,\"scope\":\"Specified scalar Dirichlet diffusion reference; not interacting kinetic swarm QSD\",\"cases\":[")?;
    let mut first_case=true;let mut checks=0usize;let mut failed=0usize;let mut updates=0usize;
    for diffusivity in [0.01_f64,1.,3.] { for length in [0.3_f64,1.,5.] { for grid in [32usize,64,128] {
        if !first_case {writeln!(w,",")?;}first_case=false;
        let dx=length/grid as f64;let r=0.25;let dt=r*dx*dx/diffusivity;
        let mut f:Vec<_>=(0..=grid).map(|i|std::f64::consts::PI/(2.*length)*(std::f64::consts::PI*i as f64/grid as f64).sin()).collect();
        f[0]=0.;f[grid]=0.;let initial_mass=f.iter().sum::<f64>()*dx;
        let rate=diffusivity*std::f64::consts::PI.powi(2)/length.powi(2);
        write!(w,"{{\"D0\":{diffusivity},\"L\":{length},\"grid_cells\":{grid},\"dx\":{dx},\"dt\":{dt},\"prediction_rate\":{rate},\"initial_mass\":{initial_mass},\"all_time_states\":[")?;
        for step in 0..=1024 {
            if step>0 {write!(w,",")?;}
            write!(w,"[")?;
            for (i,value) in f.iter().enumerate() {if i>0 {write!(w,",")?;}write!(w,"{value:.17e}")?;}
            write!(w,"]")?;
            if step<1024 {
                let mut next=vec![0.;grid+1];
                for i in 1..grid {next[i]=f[i]+r*(f[i-1]-2.*f[i]+f[i+1]);}
                f=next;updates+=1;
            }
        }
        let final_mass=f.iter().sum::<f64>()*dx;
        let measured_rate=-(final_mass/initial_mass).ln()/(1024.*dt);
        // For theta=pi/J and r=1/4, exact stencil multiplier is
        // cos²(theta/2). Thus rate/continuum = -8 log(cos(theta/2))/theta².
        // For theta<=pi/32, Taylor remainder bounds imply relative error
        // <=theta²; use this conservative explicit convergence allowance.
        let allowance=rate*(std::f64::consts::PI/grid as f64).powi(2);
        let passed=(measured_rate-rate).abs()<=allowance+1e-11*rate;
        checks+=1;failed+=usize::from(!passed);
        let normalized_shape_defect=f.iter().enumerate().map(|(i,value)| {
            let original=std::f64::consts::PI/(2.*length)*(std::f64::consts::PI*i as f64/grid as f64).sin()/initial_mass;
            (value/final_mass-original).abs()*dx
        }).sum::<f64>();
        checks+=1;failed+=usize::from(normalized_shape_defect>1e-11);
        write!(w,"],\"final_mass\":{final_mass},\"measured_decay_rate\":{measured_rate},\"rate_allowance\":{allowance},\"rate_passed\":{passed},\"normalized_shape_L1_defect\":{normalized_shape_defect},\"forced_sources\":[")?;
        for (source_index,source) in [0.2_f64,2.].into_iter().enumerate() {
            if source_index>0 {write!(w,",")?;}
            let forced:Vec<_>=(0..=grid).map(|i| {let x=i as f64*dx;source*x*(length-x)/(2.*diffusivity)}).collect();
            let maximum=(1..grid).map(|i|(diffusivity*(forced[i-1]-2.*forced[i]+forced[i+1])/(dx*dx)+source).abs()).fold(0_f64,f64::max);
            let passed=maximum<1e-9*(1.+source);
            checks+=1;failed+=usize::from(!passed);
            write!(w,"{{\"source\":{source},\"maximum_discrete_source_balance_defect\":{maximum},\"passed\":{passed},\"density\":[")?;
            for (i,value) in forced.iter().enumerate() {if i>0 {write!(w,",")?;}write!(w,"{value:.17e}")?;}
            write!(w,"]}}")?;
        }
        write!(w,"]}}")?;
    }}}
    writeln!(w,"],\"summary\":{{\"reference_time_updates\":{updates},\"comparisons\":{checks},\"comparisons_failed\":{failed},\"new_native_swarm_updates\":0}}}}")?;
    w.flush()?;if failed>0 {return Err("Reference diffusion comparison failed; evidence retained".into());}Ok(())
}
