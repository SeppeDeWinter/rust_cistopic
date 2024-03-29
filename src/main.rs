mod lda;
use ndarray::Array;
use npyz;
use rand::seq::SliceRandom;
use rand::Rng;
use std::time::Instant;

struct GenomicRegion {
    chromosome: String,
    start: usize,
    end: usize,
}

impl GenomicRegion {
    fn from_str(s: &str) -> GenomicRegion {
        let fields: Vec<&str> = s.split('\t').collect();
        GenomicRegion {
            chromosome: fields[0].to_string(),
            start: fields[1].parse().unwrap(),
            end: fields[2].parse().unwrap(),
        }
    }
}
struct Fragment {
    region: GenomicRegion,
    cell_barcode: String,
    score: Option<usize>,
}

impl Fragment {
    fn from_str(s: &str) -> Fragment {
        let fields: Vec<&str> = s.split('\t').collect();
        match fields.len() {
            4 => Fragment {
                region: GenomicRegion {
                    chromosome: fields[0].to_string(),
                    start: fields[1].parse().unwrap(),
                    end: fields[2].parse().unwrap(),
                },
                cell_barcode: fields[3].to_string(),
                score: None,
            },
            5 => Fragment {
                region: GenomicRegion {
                    chromosome: fields[0].to_string(),
                    start: fields[1].parse().unwrap(),
                    end: fields[2].parse().unwrap(),
                },
                cell_barcode: fields[3].to_string(),
                score: Some(fields[4].parse().unwrap()),
            },
            _ => panic!("Invalid number of fields"),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing variables ...");
    let D = 5_000;
    let W = 2_000_000;
    let N = 999_950;
    let n_topics = 50;
    let n_iter = 20;
    let mut nzw = Array::<f64, _>::zeros((W, n_topics));
    let mut ndz = Array::<f64, _>::zeros((D, n_topics));
    let mut nz = vec![0.0; n_topics];
    println!("Reading data ...");
    let bytes = std::fs::read("/staging/leuven/stg_00002/lcb/sdewin/PhD/rust_module/lda/WS.npy")?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let data = npy.data::<i32>()?;
    let mut WS: Vec<usize> = Vec::with_capacity(data.len() as usize);
    for x in data {
        let x = x?;
        WS.push(x as usize);
    }
    let bytes = std::fs::read("/staging/leuven/stg_00002/lcb/sdewin/PhD/rust_module/lda/DS.npy")?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let data = npy.data::<i32>()?;
    let mut DS: Vec<usize> = Vec::with_capacity(data.len() as usize);
    for x in data {
        let x = x?;
        DS.push(x as usize);
    }
    let mut ZS: Vec<usize> = vec![0; WS.len()];
    for i in 0..N {
        let w = WS[i];
        let d = DS[i];
        let z_new = i % n_topics;
        ZS[i] = z_new;
        ndz[[d, z_new]] += 1.0;
        nzw[[w, z_new]] += 1.0;
        nz[z_new] += 1.0;
    }
    let mut rands: Vec<f64> = (0..131072)
        .map(|_| rand::thread_rng().gen::<f64>())
        .collect();
    let alpha: Vec<f64> = vec![0.1; n_topics];
    let eta: Vec<f64> = vec![0.01; W];
    println!("Running LDA ...");
    let mut rng = rand::thread_rng();
    let now = Instant::now();
    for _ in 0..n_iter {
        rands.shuffle(&mut rng);
        lda::sample_topics(
            &WS, &DS, &mut ZS, &mut nzw, &mut ndz, &mut nz, &alpha, &eta, &rands,
        )
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    Ok(())
}
