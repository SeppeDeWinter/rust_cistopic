fn searchsorted(
    arr: &Vec<f64>,
    length: usize,
    value: f64) -> usize 
{
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
    WS: &Vec<usize>,
    DS: &Vec<usize>,
    ZS: &mut Vec<usize>,
    nzw: &mut Vec<Vec<f64>>,
    ndz: &mut Vec<Vec<f64>>,
    nz: &mut Vec<f64>,
    alpha: &Vec<f64>,
    eta: &Vec<f64>,
    rands: &Vec<f64>)
{
    // initialize variables
    let N                = WS.len();
    let n_rand           = rands.len();
    let n_topics         = nz.len();
    let eta_sum:f64             = eta.iter().sum();
    let mut dist_sum: Vec<f64>  = Vec::with_capacity(n_topics);

    // actual algorithm
    for i in 0..N {
        let w = WS[i];
        let d = DS[i];
        let z = ZS[i];
        nzw[w][z] -= 1.0;
        ndz[d][z] -= 1.0;
        nz[z]     -= 1.0;
        dist_sum = (0..n_topics).map(|k| {
            (nzw[w][k] + eta[w]) / (nz[k] + eta_sum) * (ndz[d][k] + alpha[k])
        }).collect();
        let mut acc = 0.0;
        for x in &mut dist_sum {
            acc += *x;
            *x = acc;
        }
        let dist_cum = dist_sum[n_topics - 1];
        let r = rands[i % n_rand] * dist_cum;
        let z_new = searchsorted(&dist_sum, n_topics, r);
        ZS[i] = z_new;
        nzw[w][z_new] += 1.0;
        ndz[d][z_new] += 1.0;
        nz[z_new]     += 1.0;
    }
} 
