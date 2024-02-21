mod lda;
use npyz;
use rand::Rng;
use rand::seq::SliceRandom;

struct GenomicRegion {
    chromosome: String,
    start: usize,
    end: usize
}

impl GenomicRegion{
    fn from_str(s: &str) -> GenomicRegion {
        let fields: Vec<&str> = s.split('\t').collect();
        GenomicRegion {
            chromosome: fields[0].to_string(),
            start: fields[1].parse().unwrap(),
            end: fields[2].parse().unwrap()
        }
    }
}
struct Fragment {
    region: GenomicRegion,
    cell_barcode: String,
    score: Option<usize>
}

impl Fragment {
    fn from_str(s: &str) -> Fragment {
        let fields: Vec<&str> = s.split('\t').collect();
        match fields.len() {
            4 => Fragment {
                region: GenomicRegion {
                    chromosome: fields[0].to_string(),
                    start: fields[1].parse().unwrap(),
                    end: fields[2].parse().unwrap()
                },
                cell_barcode: fields[3].to_string(),
                score: None
            },
            5 => Fragment {
                region: GenomicRegion {
                    chromosome: fields[0].to_string(),
                    start: fields[1].parse().unwrap(),
                    end: fields[2].parse().unwrap()
                },
                cell_barcode: fields[3].to_string(),
                score: Some(fields[4].parse().unwrap())
            },
            _ => panic!("Invalid number of fields")
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>>{
    println!("Initializing variables ...");
    let D = 5_000;
    let W = 2_000_000;
    let N = 999_950;
    let n_topics = 50;
    let n_iter = 20;
    let mut nzw = vec![vec![0.0; n_topics]; W];
    let mut ndz = vec![vec![0.0; n_topics]; D];
    let mut nz = vec![0.0; n_topics];
    println!("Reading data ...");
    let bytes = std::fs::read("/staging/leuven/stg_00002/lcb/sdewin/PhD/rust_module/lda/WS.npy")?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let mut WS: Vec<usize> = Vec::new();
    for x in npy.data::<u32>()? {
        let x = x?;
        WS.push(x as usize);
    }
    let bytes = std::fs::read("/staging/leuven/stg_00002/lcb/sdewin/PhD/rust_module/lda/DS.npy")?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let mut DS: Vec<usize> = Vec::new();
    for x in npy.data::<u32>()? {
        let x = x?;
        DS.push(x as usize);
    }
    let mut ZS: Vec<usize> = vec![0; WS.len()];
    for i in 0..N {
        let w = WS[i];
        let d = DS[i];
        let z_new = i % n_topics;
        ZS[i] = z_new;
        ndz[d][z_new] += 1.0;
        nzw[w][z_new] += 1.0;
        nz[z_new] += 1.0;
    }
    let mut rands: Vec<f64> = (0..131072).map(|_| rand::thread_rng().gen::<f64>()).collect();
    let alpha: Vec<f64> = vec![0.1; n_topics];
    let eta: Vec<f64> = vec![0.01; W];
    println!("Running LDA ...");
    let mut rng = rand::thread_rng();
    for _ in 0..n_iter {
        rands.shuffle(&mut rng);
        lda::sample_topics(
            &WS,
            &DS,
            &mut ZS,
            &mut nzw,
            &mut ndz,
            &mut nz,
            &alpha,
            &eta,
            &rands
        )
    }

    Ok(())
}
