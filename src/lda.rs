use ndarray::Array2;

fn searchsorted(arr: &[f64], length: usize, value: f64) -> usize {
    let mut imin = 0;
    let mut imax = length;
    while imin < imax {
        let imid = imin + ((imax - imin) >> 1);
        if value > arr[imid] {
            imin = imid + 1;
        } else {
            imax = imid;
        }
    }
    imin
}

pub fn sample_topics(
    WS: &[usize],
    DS: &[usize],
    ZS: &mut [usize],
    nzw: &mut Array2<f64>,
    ndz: &mut Array2<f64>,
    nz: &mut [f64],
    alpha: &[f64],
    eta: &[f64],
    rands: &[f64],
) {
    // initialize variables
    let N = WS.len();
    let n_rand = rands.len();
    let n_topics = nz.len();
    let eta_sum: f64 = eta.iter().sum();

    // create dist_sum as a Vec<f64> of length n_topics with zeros
    let mut dist_sum: Vec<f64> = vec![0.0; n_topics];

    // actual algorithm
    for i in 0..N {
        let w = WS[i];
        let d = DS[i];
        let z = ZS[i];

        nzw[[w, z]] -= 1.0;
        ndz[[d, z]] -= 1.0;
        nz[z] -= 1.0;

        // run this in parallel?
        for k in 0..n_topics {
            dist_sum[k] = (nzw[[w, k]] + eta[w]) / (nz[k] + eta_sum) * (ndz[[d, k]] + alpha[k]);
        }

        // Calculating cumulative sum with into_iter will reuse the memory of dist_sum.
        // See: https://users.rust-lang.org/t/inplace-cumulative-sum-using-iterator/56532/5
        dist_sum = dist_sum
            .into_iter()
            .scan(0.0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        let r = rands[i % n_rand] * dist_sum[n_topics - 1];
        let z_new = searchsorted(&dist_sum, n_topics, r);
        ZS[i] = z_new;
        nzw[[w, z_new]] += 1.0;
        ndz[[d, z_new]] += 1.0;
        nz[z_new] += 1.0;
    }
}
