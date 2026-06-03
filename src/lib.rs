use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;

/// Error type for spectral graph operations
#[derive(Debug, Clone, PartialEq)]
pub enum SgError {
    NullInput,
    AllocationFailed,
    SingularMatrix,
    NoConvergence,
    InvalidParam,
    Disconnected,
}

impl std::fmt::Display for SgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SgError::NullInput => write!(f, "NULL pointer"),
            SgError::AllocationFailed => write!(f, "allocation failure"),
            SgError::SingularMatrix => write!(f, "singular matrix"),
            SgError::NoConvergence => write!(f, "no convergence"),
            SgError::InvalidParam => write!(f, "invalid parameter"),
            SgError::Disconnected => write!(f, "disconnected graph"),
        }
    }
}

/// Graph represented as an adjacency list with weighted edges
#[derive(Debug, Clone)]
pub struct SpectralGraph {
    /// Number of vertices
    pub n: usize,
    /// Adjacency list: neighbors[edge_idx] = (neighbor, weight)
    /// Stored as (col_idx, weight) pairs per vertex
    adj: Vec<Vec<(usize, f64)>>,
    directed: bool,
    finalized: bool,
    /// Pending edges during construction
    pending_edges: Vec<(usize, usize, f64)>,
}

impl SpectralGraph {
    /// Create a new graph with n vertices
    pub fn new(n: usize, directed: bool) -> Self {
        SpectralGraph {
            n,
            adj: vec![vec![]; n],
            directed,
            finalized: false,
            pending_edges: Vec::new(),
        }
    }

    /// Add an edge (u, v) with weight w
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) -> Result<(), SgError> {
        if u >= self.n || v >= self.n {
            return Err(SgError::InvalidParam);
        }
        if self.finalized {
            return Err(SgError::InvalidParam);
        }
        self.pending_edges.push((u, v, w));
        if !self.directed && u != v {
            self.pending_edges.push((v, u, w));
        }
        Ok(())
    }

    /// Finalize the graph (build adjacency list from pending edges)
    pub fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        self.adj = vec![vec![]; self.n];
        for (u, v, w) in &self.pending_edges {
            self.adj[*u].push((*v, *w));
        }
        self.finalized = true;
    }

    // ── Builder helpers ─────────────────────────────────────

    /// Build a complete graph
    pub fn build_complete(&mut self, w: f64) {
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                self.add_edge(i, j, w).ok();
            }
        }
        self.finalize();
    }

    /// Build a path graph
    pub fn build_path(&mut self, w: f64) {
        for i in 0..self.n.saturating_sub(1) {
            self.add_edge(i, i + 1, w).ok();
        }
        self.finalize();
    }

    /// Build a cycle graph
    pub fn build_cycle(&mut self, w: f64) {
        for i in 0..self.n {
            self.add_edge(i, (i + 1) % self.n, w).ok();
        }
        self.finalize();
    }

    /// Build a star graph (center = vertex 0)
    pub fn build_star(&mut self, w: f64) {
        for i in 1..self.n {
            self.add_edge(0, i, w).ok();
        }
        self.finalize();
    }

    /// Build a 2D grid graph
    pub fn build_grid2d(&mut self, rows: usize, cols: usize, w: f64) -> Result<(), SgError> {
        if rows * cols != self.n {
            return Err(SgError::InvalidParam);
        }
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                if c + 1 < cols {
                    self.add_edge(v, v + 1, w).ok();
                }
                if r + 1 < rows {
                    self.add_edge(v, v + cols, w).ok();
                }
            }
        }
        self.finalize();
        Ok(())
    }

    // ── Graph properties ────────────────────────────────────

    /// Get the degree of vertex v
    pub fn degree(&self, v: usize) -> usize {
        if v >= self.n {
            return 0;
        }
        self.adj[v].len()
    }

    /// Get the weighted degree of vertex v
    pub fn weighted_degree(&self, v: usize) -> f64 {
        if v >= self.n {
            return 0.0;
        }
        self.adj[v].iter().map(|(_, w)| w).sum()
    }

    /// Count edges
    pub fn edge_count(&self) -> usize {
        let total: usize = (0..self.n).map(|i| self.adj[i].len()).sum();
        if self.directed { total } else { total / 2 }
    }

    /// Check connectivity via BFS from vertex 0
    pub fn is_connected(&self) -> bool {
        if self.n <= 1 {
            return true;
        }
        let mut visited = vec![false; self.n];
        let mut queue = VecDeque::new();
        queue.push_back(0);
        visited[0] = true;
        let mut count = 1;
        while let Some(v) = queue.pop_front() {
            for &(u, _) in &self.adj[v] {
                if !visited[u] {
                    visited[u] = true;
                    count += 1;
                    queue.push_back(u);
                }
            }
        }
        count == self.n
    }

    /// Check if graph is regular, returns degree if so
    pub fn is_regular(&self) -> Option<usize> {
        if self.n == 0 {
            return Some(0);
        }
        let d = self.degree(0);
        for i in 1..self.n {
            if self.degree(i) != d {
                return None;
            }
        }
        Some(d)
    }

    // ── Matrix construction ─────────────────────────────────

    /// Get adjacency matrix as DMatrix
    pub fn adjacency_matrix(&self) -> DMatrix<f64> {
        let mut data = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            for &(j, w) in &self.adj[i] {
                data[i * self.n + j] = w;
            }
        }
        DMatrix::from_row_slice(self.n, self.n, &data)
    }

    /// Get Laplacian matrix L = D - A
    pub fn laplacian(&self) -> DMatrix<f64> {
        let mut data = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            let mut deg = 0.0;
            for &(j, w) in &self.adj[i] {
                data[i * self.n + j] = -w;
                deg += w;
            }
            data[i * self.n + i] = deg;
        }
        DMatrix::from_row_slice(self.n, self.n, &data)
    }

    /// Get normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2}
    pub fn normalized_laplacian(&self) -> DMatrix<f64> {
        let mut deg = vec![0.0; self.n];
        for i in 0..self.n {
            for &(j, w) in &self.adj[i] {
                deg[i] += w;
            }
        }
        let d_inv_sqrt: Vec<f64> = deg
            .iter()
            .map(|&d| if d > 1e-15 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        let mut data = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            data[i * self.n + i] = 1.0; // identity
            for &(j, w) in &self.adj[i] {
                data[i * self.n + j] -= d_inv_sqrt[i] * w * d_inv_sqrt[j];
            }
        }
        DMatrix::from_row_slice(self.n, self.n, &data)
    }

    // ── Eigendecomposition ──────────────────────────────────

    /// QR-based eigendecomposition using Wilkinson shifts
    /// Returns (eigenvalues sorted ascending, eigenvectors as columns)
    pub fn eigendecompose(matrix: &DMatrix<f64>) -> Result<(Vec<f64>, DMatrix<f64>), SgError> {
        let n = matrix.nrows();
        if n == 0 {
            return Err(SgError::NullInput);
        }

        // Check for zero matrix
        let all_zero = matrix.iter().all(|&v| v.abs() < 1e-15);

        let mut a = matrix.clone();
        let mut q_total = DMatrix::identity(n, n);

        let max_iter = 200 + n * 10;
        let mut iterations = 0;

        for _ in 0..max_iter {
            iterations += 1;

            // Wilkinson shift
            if n >= 2 {
                let a_val = a[(n - 2, n - 2)];
                let b_val = a[(n - 2, n - 1)];
                let c_val = a[(n - 1, n - 2)];
                let d_val = a[(n - 1, n - 1)];
                let tr = a_val + d_val;
                let det = a_val * d_val - b_val * c_val;
                let disc = (tr * tr / 4.0 - det).abs().sqrt();
                let mu1 = tr / 2.0 + disc;
                let mu2 = tr / 2.0 - disc;
                let mu = if (mu1 - d_val).abs() < (mu2 - d_val).abs() {
                    mu1
                } else {
                    mu2
                };
                // Apply shift
                for i in 0..n {
                    a[(i, i)] -= mu;
                }

                // QR step
                let qr = a.clone().qr();
                let q_step = qr.q();
                let r = qr.r();
                a = &r * &q_step;

                // Re-add shift
                for i in 0..n {
                    a[(i, i)] += mu;
                }

                // Accumulate
                q_total = &q_total * &q_step;
            } else {
                break;
            }

            // Check convergence (off-diagonal norm)
            let mut off = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        off += a[(i, j)] * a[(i, j)];
                    }
                }
            }
            if off.sqrt() < 1e-12 {
                break;
            }
        }

        // Extract eigenvalues from diagonal
        let mut eigenvalues: Vec<(f64, usize)> = (0..n).map(|i| (a[(i, i)], i)).collect();
        eigenvalues.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let sorted_eigenvalues: Vec<f64> = eigenvalues.iter().map(|(v, _)| *v).collect();

        // Reorder eigenvectors
        let n2 = q_total.ncols();
        let mut sorted_eigvecs = DMatrix::zeros(n, n2);
        for (new_idx, (_, old_idx)) in eigenvalues.iter().enumerate() {
            for r in 0..n {
                sorted_eigvecs[(r, new_idx)] = q_total[(r, *old_idx)];
            }
        }

        Ok((sorted_eigenvalues, sorted_eigvecs))
    }

    /// Power iteration for dominant eigenvector
    pub fn power_iteration(
        matrix: &DMatrix<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<(f64, DVector<f64>), SgError> {
        let n = matrix.nrows();
        if n == 0 {
            return Err(SgError::NullInput);
        }

        // Bug #1 fix: Check if matrix is all-zero
        let all_zero = matrix.iter().all(|&v| v.abs() < 1e-15);
        if all_zero {
            return Err(SgError::SingularMatrix);
        }

        // Random initial vector
        let mut v = DVector::from_fn(n, |i, _| ((i as f64 * 1.234 + 0.567) % 1.0));
        let norm = v.norm();
        if norm > 1e-15 {
            v /= norm;
        }

        for _ in 0..max_iter {
            let y = matrix * &v;
            let eigenvalue = v.dot(&y);
            let y_norm = y.norm();
            if y_norm < 1e-15 {
                break;
            }
            let y_normalized = &y / y_norm;
            let diff: f64 = (&v - &y_normalized)
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            v = y_normalized;
            if diff < tol {
                return Ok((eigenvalue, v));
            }
        }

        Ok((v.dot(&(matrix * &v)), v))
    }

    // ── Core spectral analyses ──────────────────────────────

    /// Compute the Fiedler value (algebraic connectivity = λ₂ of Laplacian)
    /// Returns the algebraic value (NOT absolute)
    pub fn fiedler_value(&self) -> Result<f64, SgError> {
        let L = self.laplacian();
        let (eigenvalues, _) = Self::eigendecompose(&L)?;
        if eigenvalues.len() < 2 {
            return Err(SgError::InvalidParam);
        }
        Ok(eigenvalues[1])
    }

    /// Compute the Fiedler vector
    pub fn fiedler_vector(&self) -> Result<DVector<f64>, SgError> {
        let L = self.laplacian();
        let (eigenvalues, eigvecs) = Self::eigendecompose(&L)?;
        if eigenvalues.len() < 2 {
            return Err(SgError::InvalidParam);
        }
        Ok(DVector::from_fn(self.n, |i, _| eigvecs[(i, 1)]))
    }

    /// Compute the Cheeger constant using sweep cut with actual edge weights
    pub fn cheeger_constant(&self) -> Result<f64, SgError> {
        let fiedler = self.fiedler_vector()?;
        let n = self.n;

        // Sort vertices by Fiedler vector value
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| fiedler[a].partial_cmp(&fiedler[b]).unwrap());

        // Sweep cut
        let mut best_cond = 1.0;
        for k in 1..n {
            let cond = self.conductance_of_set(&idx[..k]);
            if cond < best_cond {
                best_cond = cond;
            }
        }
        Ok(best_cond)
    }

    /// Conductance of a set S
    fn conductance_of_set(&self, s: &[usize]) -> f64 {
        if s.is_empty() || s.len() >= self.n {
            return 1.0;
        }
        let in_s: Vec<bool> = {
            let mut v = vec![false; self.n];
            for &i in s {
                v[i] = true;
            }
            v
        };

        let mut vol_s = 0.0;
        let mut vol_comp = 0.0;
        let mut cut = 0.0;

        for v in 0..self.n {
            // Bug #4 fix: Use weighted degree
            let deg: f64 = self.adj[v].iter().map(|(_, w)| w).sum();
            if in_s[v] {
                vol_s += deg;
                for &(u, w) in &self.adj[v] {
                    if !in_s[u] {
                        cut += w;
                    }
                }
            } else {
                vol_comp += deg;
            }
        }

        let vol_min = vol_s.min(vol_comp);
        if vol_min > 0.0 {
            cut / vol_min
        } else {
            1.0
        }
    }

    /// Eigenvector centrality with max(1, max_c) denominator
    pub fn centrality(&self) -> Vec<f64> {
        let A = self.adjacency_matrix();
        let power_result = Self::power_iteration(&A, 1000, 1e-10);

        let mut cent: Vec<f64> = match power_result {
            Ok((_, vec)) => vec.iter().copied().collect(),
            Err(_) => {
                // Fallback: degree centrality
                (0..self.n).map(|i| self.degree(i) as f64).collect()
            }
        };

        // Take absolute values
        for v in &mut cent {
            *v = v.abs();
        }

        // Bug #2 fix: Normalize with max(1, max_centrality) to handle isolated vertices
        let max_c = cent.iter().cloned().fold(0.0f64, f64::max);
        let norm = if max_c > 1e-15 { max_c } else { 1.0 };
        for v in &mut cent {
            *v /= norm;
        }

        cent
    }

    /// Mixing time. Returns None for disconnected graphs.
    pub fn mixing_time(&self) -> Option<f64> {
        // Bug #3 fix: Check connectivity first
        if !self.is_connected() {
            return None;
        }
        if self.n < 2 {
            return Some(0.0);
        }

        let Lnorm = self.normalized_laplacian();
        let (eigenvalues, _) = Self::eigendecompose(&Lnorm).ok()?;

        let lambda2 = eigenvalues[1]; // second smallest eigenvalue of normalized Laplacian

        if lambda2 > 1e-15 {
            Some((1.0 / lambda2) * (self.n as f64 / 0.01).ln())
        } else {
            None // effectively infinite
        }
    }

    /// Expander quality with correct orientation
    pub fn expander_quality(&self) -> Result<f64, SgError> {
        let cheeger = self.cheeger_constant()?;
        let regular = self.is_regular();

        if let Some(d) = regular {
            if d >= 2 {
                let ramanujan_bound = 2.0 * ((d - 1) as f64).sqrt();

                let A = self.adjacency_matrix();
                let (eigenvalues, _) = Self::eigendecompose(&A)?;
                if eigenvalues.len() < 2 {
                    return Ok(0.0);
                }

                // Find max |λ| among non-dominant eigenvalues
                let n = eigenvalues.len();
                let max_abs: f64 = (0..n - 1)
                    .map(|i| eigenvalues[i].abs())
                    .fold(0.0f64, f64::max);

                // Bug #5 fix: ratio = bound / max_abs, so higher = better
                return if max_abs > 1e-15 {
                    Ok(ramanujan_bound / max_abs)
                } else {
                    Ok(0.0)
                };
            }
        }
        Ok(0.0)
    }

    /// Spectral gap from adjacency matrix (algebraic λ₁ - λ₂)
    pub fn spectral_gap(&self) -> Result<f64, SgError> {
        let A = self.adjacency_matrix();
        let (eigenvalues, _) = Self::eigendecompose(&A)?;
        if eigenvalues.len() < 2 {
            return Err(SgError::InvalidParam);
        }
        let n = eigenvalues.len();
        // Bug #6 fix: algebraic diff (largest - second largest), NOT absolute
        let lambda1 = eigenvalues[n - 1]; // largest
        let lambda2 = eigenvalues[n - 2]; // second largest
        Ok(lambda1 - lambda2)
    }

    /// Full eigendecompose: returns (dominant eigenvalue, dominant eigenvector)
    /// Returns error for zero matrices
    pub fn eigendecompose_public(&self) -> Result<(f64, DVector<f64>), SgError> {
        let A = self.adjacency_matrix();
        Self::power_iteration(&A, 1000, 1e-10)
    }

    /// Compute the spectral gap from the normalized Laplacian
    pub fn normalized_spectral_gap(&self) -> Result<f64, SgError> {
        let Lnorm = self.normalized_laplacian();
        let (eigenvalues, _) = Self::eigendecompose(&Lnorm)?;
        if eigenvalues.len() < 2 {
            return Err(SgError::InvalidParam);
        }
        Ok(eigenvalues[1])
    }

    /// Edge connectivity (approximate: minimum degree)
    pub fn edge_connectivity(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let mut min_deg = self.degree(0);
        for i in 1..self.n {
            let d = self.degree(i);
            if d < min_deg {
                min_deg = d;
            }
        }
        min_deg as f64
    }

    /// Spectral clustering into k clusters
    pub fn spectral_cluster(&self, k: usize) -> Result<Vec<usize>, SgError> {
        if k == 0 || k > self.n {
            return Err(SgError::InvalidParam);
        }
        if k == 1 {
            return Ok(vec![0; self.n]);
        }

        let L = self.laplacian();
        let (_, eigvecs) = Self::eigendecompose(&L)?;

        let mut cluster_ids = vec![0; self.n];

        if k == 2 {
            for i in 0..self.n {
                cluster_ids[i] = if eigvecs[(i, 1)] >= 0.0 { 0 } else { 1 };
            }
        } else {
            for i in 0..self.n {
                let val = eigvecs[(i, 1)];
                let mut best_c = 0;
                let mut min_dist = f64::MAX;
                for c in 0..k {
                    let center = -1.0 + 2.0 * c as f64 / (k - 1) as f64;
                    let dist = (val - center).abs();
                    if dist < min_dist {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                cluster_ids[i] = best_c;
            }
        }

        Ok(cluster_ids)
    }

    /// Count articulation points
    pub fn count_articulation_points(&self) -> usize {
        if self.n <= 2 {
            return 0;
        }

        let mut visited = vec![false; self.n];
        let mut disc = vec![0i32; self.n];
        let mut low = vec![0i32; self.n];
        let mut parent = vec![-1i32; self.n];
        let mut ap = vec![false; self.n];
        let mut time = 0i32;

        for start in 0..self.n {
            if visited[start] {
                continue;
            }
            // Iterative DFS
            let mut stack = vec![(start, 0usize, false)];

            while let Some((v, adj_idx, processed)) = stack.last_mut() {
                let v = *v;
                if !visited[v] {
                    visited[v] = true;
                    disc[v] = time;
                    low[v] = time;
                    time += 1;
                }

                if *adj_idx < self.adj[v].len() {
                    let (u, _) = self.adj[v][*adj_idx];
                    *adj_idx += 1;
                    if !visited[u] {
                        parent[u] = v as i32;
                        stack.push((u, 0, false));
                    } else if u as i32 != parent[v] {
                        low[v] = low[v].min(disc[u]);
                    }
                } else {
                    // All neighbors processed
                    if parent[v] != -1 {
                        let p = parent[v] as usize;
                        low[p] = low[p].min(low[v]);
                        if parent[p] != -1 && low[v] >= disc[p] {
                            ap[p] = true;
                        }
                    } else {
                        // Root: AP if 2+ children
                        let children = self.adj[v]
                            .iter()
                            .filter(|&&(u, _)| parent[u] == v as i32)
                            .count();
                        if children > 1 {
                            ap[v] = true;
                        }
                    }
                    stack.pop();

                    // Propagate low values to parent
                    if let Some((pv, _, _)) = stack.last_mut() {
                        let p = *pv;
                        if parent[v as usize] == p as i32 {
                            low[p] = low[p].min(low[v as usize]);
                        }
                    }
                }
            }
        }

        ap.iter().filter(|&&x| x).count()
    }

    /// Compute robustness score
    pub fn robustness(&self) -> f64 {
        if self.n <= 1 {
            return 1.0;
        }
        let connected = self.is_connected();
        if !connected {
            return 0.0;
        }

        let fiedler = self.fiedler_value().unwrap_or(0.0);
        let conn_score = if fiedler > 2.0 { 1.0 } else { fiedler / 2.0 };
        let ap_count = self.count_articulation_points() as f64;
        let ap_penalty = ap_count / self.n as f64;
        let score = conn_score * (1.0 - ap_penalty);
        score.max(0.0).min(1.0)
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;
    const EPS_LOOSE: f64 = 1e-3;

    // ── 1. Graph Construction ──────────────────────────────

    #[test]
    fn graph_create_destroy() {
        let g = SpectralGraph::new(5, false);
        assert_eq!(g.n, 5);
    }

    #[test]
    fn graph_create_zero() {
        let g = SpectralGraph::new(0, false);
        assert_eq!(g.n, 0);
    }

    #[test]
    fn graph_add_edge_basic() {
        let mut g = SpectralGraph::new(4, false);
        assert!(g.add_edge(0, 1, 1.0).is_ok());
        g.finalize();
    }

    #[test]
    fn graph_add_edge_out_of_bounds() {
        let mut g = SpectralGraph::new(3, false);
        assert_eq!(g.add_edge(0, 5, 1.0), Err(SgError::InvalidParam));
    }

    #[test]
    fn graph_build_path() {
        let mut g = SpectralGraph::new(5, false);
        g.build_path(1.0);
        assert_eq!(g.edge_count(), 4);
        assert!(g.is_connected());
    }

    #[test]
    fn graph_build_cycle() {
        let mut g = SpectralGraph::new(6, false);
        g.build_cycle(1.0);
        assert_eq!(g.edge_count(), 6);
        assert!(g.is_connected());
    }

    #[test]
    fn graph_build_complete() {
        let mut g = SpectralGraph::new(4, false);
        g.build_complete(1.0);
        assert_eq!(g.edge_count(), 6); // n(n-1)/2
        assert!(g.is_connected());
    }

    #[test]
    fn graph_build_star() {
        let mut g = SpectralGraph::new(5, false);
        g.build_star(1.0);
        assert_eq!(g.edge_count(), 4);
        assert!(g.is_connected());
        assert_eq!(g.degree(0), 4); // center
        assert_eq!(g.degree(1), 1); // leaf
    }

    #[test]
    fn graph_build_grid2d() {
        let mut g = SpectralGraph::new(6, false); // 2x3
        assert!(g.build_grid2d(2, 3, 1.0).is_ok());
        assert!(g.is_connected());
        assert_eq!(g.edge_count(), 7); // 2*(3-1) + 3*(2-1) = 7
    }

    #[test]
    fn graph_build_grid2d_bad_size() {
        let mut g = SpectralGraph::new(7, false);
        assert_eq!(g.build_grid2d(2, 3, 1.0), Err(SgError::InvalidParam));
    }

    // ── 2. Matrix Operations ───────────────────────────────

    #[test]
    fn adjacency_matrix_complete() {
        let mut g = SpectralGraph::new(3, false);
        g.build_complete(1.0);
        let a = g.adjacency_matrix();
        assert!((a[(0, 1)] - 1.0).abs() < EPS);
        assert!((a[(1, 0)] - 1.0).abs() < EPS);
        assert!((a[(0, 0)] - 0.0).abs() < EPS);
    }

    #[test]
    fn laplacian_complete() {
        let mut g = SpectralGraph::new(3, false);
        g.build_complete(1.0);
        let l = g.laplacian();
        assert!((l[(0, 0)] - 2.0).abs() < EPS);
        assert!((l[(0, 1)] - (-1.0)).abs() < EPS);
        // Row sums = 0
        for i in 0..3 {
            let rs: f64 = (0..3).map(|j| l[(i, j)]).sum();
            assert!(rs.abs() < EPS);
        }
    }

    #[test]
    fn normalized_laplacian_complete() {
        let mut g = SpectralGraph::new(4, false);
        g.build_complete(1.0);
        let ln = g.normalized_laplacian();
        assert!((ln[(0, 0)] - 1.0).abs() < EPS);
        assert!((ln[(0, 1)] - (-1.0 / 3.0)).abs() < EPS_LOOSE);
    }

    #[test]
    fn matrix_trace() {
        let mut g = SpectralGraph::new(4, false);
        g.build_complete(1.0);
        let l = g.laplacian();
        let trace: f64 = (0..4).map(|i| l[(i, i)]).sum();
        assert!((trace - 12.0).abs() < EPS);
    }

    // ── 3. Graph Properties ────────────────────────────────

    #[test]
    fn graph_degree() {
        let mut g = SpectralGraph::new(5, false);
        g.build_star(1.0);
        assert_eq!(g.degree(0), 4);
        assert_eq!(g.degree(4), 1);
        assert_eq!(g.degree(99), 0);
    }

    #[test]
    fn graph_edge_count() {
        let mut g = SpectralGraph::new(5, false);
        g.build_cycle(1.0);
        assert_eq!(g.edge_count(), 5);
    }

    #[test]
    fn graph_is_connected_path() {
        let mut g = SpectralGraph::new(4, false);
        g.build_path(1.0);
        assert!(g.is_connected());
    }

    #[test]
    fn graph_is_connected_disconnected() {
        let mut g = SpectralGraph::new(4, false);
        g.finalize(); // no edges
        assert!(!g.is_connected());
    }

    #[test]
    fn graph_is_regular_cycle() {
        let mut g = SpectralGraph::new(6, false);
        g.build_cycle(1.0);
        assert_eq!(g.is_regular(), Some(2));
    }

    #[test]
    fn graph_is_regular_star_not() {
        let mut g = SpectralGraph::new(5, false);
        g.build_star(1.0);
        assert!(g.is_regular().is_none());
    }

    // ── 4. Eigendecomposition ──────────────────────────────

    #[test]
    fn power_iteration_complete() {
        let mut g = SpectralGraph::new(3, false);
        g.build_complete(1.0);
        let a = g.adjacency_matrix();
        let (ev, _) = SpectralGraph::power_iteration(&a, 100, 1e-10).unwrap();
        assert!((ev - 2.0).abs() < EPS_LOOSE);
    }

    #[test]
    fn eigendecompose_identity() {
        let m = DMatrix::identity(3, 3);
        let (eigenvalues, _) = SpectralGraph::eigendecompose(&m).unwrap();
        for i in 0..3 {
            assert!((eigenvalues[i] - 1.0).abs() < EPS_LOOSE);
        }
    }

    #[test]
    fn eigendecompose_laplacian_path() {
        let mut g = SpectralGraph::new(4, false);
        g.build_path(1.0);
        let l = g.laplacian();
        let (eigenvalues, _) = SpectralGraph::eigendecompose(&l).unwrap();
        assert!(eigenvalues[0].abs() < EPS_LOOSE);
        assert!(eigenvalues[1] > EPS);
    }

    #[test]
    fn eigendecompose_empty_fails() {
        let m = DMatrix::zeros(0, 0);
        assert!(SpectralGraph::eigendecompose(&m).is_err());
    }

    #[test]
    fn power_iteration_null() {
        let m = DMatrix::zeros(0, 0);
        assert!(SpectralGraph::power_iteration(&m, 100, 1e-10).is_err());
    }

    // ── 5. Fiedler Value ───────────────────────────────────

    #[test]
    fn fiedler_complete_graph() {
        let mut g = SpectralGraph::new(5, false);
        g.build_complete(1.0);
        let f = g.fiedler_value().unwrap();
        assert!((f - 5.0).abs() < EPS_LOOSE);
    }

    #[test]
    fn fiedler_path_graph() {
        let mut g = SpectralGraph::new(4, false);
        g.build_path(1.0);
        let f = g.fiedler_value().unwrap();
        assert!(f > 0.0 && f < 1.0);
    }

    #[test]
    fn fiedler_cycle_graph() {
        let mut g = SpectralGraph::new(6, false);
        g.build_cycle(1.0);
        let f = g.fiedler_value().unwrap();
        let expected = 2.0 - 2.0 * (2.0 * std::f64::consts::PI / 6.0).cos();
        assert!((f - expected).abs() < EPS_LOOSE);
    }

    #[test]
    fn fiedler_star_graph() {
        let mut g = SpectralGraph::new(5, false);
        g.build_star(1.0);
        let f = g.fiedler_value().unwrap();
        assert!((f - 1.0).abs() < EPS_LOOSE);
    }

    #[test]
    fn fiedler_vector_orthogonal_to_ones() {
        let mut g = SpectralGraph::new(4, false);
        g.build_cycle(1.0);
        let v = g.fiedler_vector().unwrap();
        let sum: f64 = v.iter().sum();
        assert!(sum.abs() < EPS_LOOSE);
    }

    // ── 6. Cheeger Constant ────────────────────────────────

    #[test]
    fn cheeger_complete() {
        let mut g = SpectralGraph::new(4, false);
        g.build_complete(1.0);
        let h = g.cheeger_constant().unwrap();
        assert!(h > 0.0);
    }

    #[test]
    fn cheeger_path() {
        let mut g = SpectralGraph::new(4, false);
        g.build_path(1.0);
        let h = g.cheeger_constant().unwrap();
        assert!(h > 0.0 && h < 1.0);
    }

    #[test]
    fn cheeger_vs_fiedler_bound() {
        let mut g = SpectralGraph::new(8, false);
        g.build_cycle(1.0);
        let f = g.fiedler_value().unwrap();
        let h = g.cheeger_constant().unwrap();
        let upper = (2.0 * f).sqrt() + EPS_LOOSE;
        assert!(h <= upper);
    }

    // ── 7. Mixing Time ─────────────────────────────────────

    #[test]
    fn mixing_complete() {
        let mut g = SpectralGraph::new(5, false);
        g.build_complete(1.0);
        let m = g.mixing_time().unwrap();
        assert!(m < 10.0);
    }

    #[test]
    fn mixing_path_slow() {
        let mut g = SpectralGraph::new(20, false);
        g.build_path(1.0);
        let m = g.mixing_time().unwrap();
        assert!(m > 0.0);
    }

    #[test]
    fn mixing_disconnected_returns_none() {
        let mut g = SpectralGraph::new(6, false);
        g.add_edge(0, 1, 1.0).ok();
        g.add_edge(1, 2, 1.0).ok();
        g.add_edge(3, 4, 1.0).ok();
        g.add_edge(4, 5, 1.0).ok();
        g.finalize();
        assert!(!g.is_connected());
        assert!(g.mixing_time().is_none());
    }

    #[test]
    fn mixing_time_decreases_with_connectivity() {
        let mut g1 = SpectralGraph::new(6, false);
        g1.build_path(1.0);
        let m1 = g1.mixing_time().unwrap();

        let mut g2 = SpectralGraph::new(6, false);
        g2.build_complete(1.0);
        let m2 = g2.mixing_time().unwrap();

        assert!(m2 < m1);
    }

    // ── 8. Eigenvector Centrality ──────────────────────────

    #[test]
    fn centrality_star_center_highest() {
        let mut g = SpectralGraph::new(5, false);
        g.build_star(1.0);
        let c = g.centrality();
        assert!(c[0] >= c[1]);
        assert!((c[0] - 1.0).abs() < EPS_LOOSE);
    }

    #[test]
    fn centrality_complete_equal() {
        let mut g = SpectralGraph::new(4, false);
        g.build_complete(1.0);
        let c = g.centrality();
        for i in 1..4 {
            assert!((c[i] - c[0]).abs() < EPS_LOOSE);
        }
    }

    #[test]
    fn centrality_normalized() {
        let mut g = SpectralGraph::new(6, false);
        g.build_cycle(1.0);
        let c = g.centrality();
        for v in &c {
            assert!(*v >= 0.0 && *v <= 1.0 + EPS);
        }
    }

    #[test]
    fn centrality_isolated_vertices() {
        // 4-node graph: 0-1 connected, 2 and 3 isolated
        let mut g = SpectralGraph::new(4, false);
        g.add_edge(0, 1, 1.0).ok();
        g.finalize();
        let c = g.centrality();
        for v in &c {
            assert!(*v >= 0.0 && *v <= 1.0 + EPS);
        }
    }

    #[test]
    fn centrality_all_isolated() {
        let mut g = SpectralGraph::new(3, false);
        g.finalize();
        let c = g.centrality();
        for v in &c {
            assert!(v.abs() < EPS);
        }
    }

    // ── 9. Expander Quality ────────────────────────────────

    #[test]
    fn expander_complete() {
        let mut g = SpectralGraph::new(4, false);
        g.build_complete(1.0);
        let q = g.expander_quality().unwrap();
        // K_4 is 3-regular, Ramanujan bound = 2√2 ≈ 2.828
        // eigenvalues of K_4: [3, -1, -1, -1], max_abs = 1
        // quality = 2√2 / 1 = 2√2 ≈ 2.828
        assert!(q >= 1.0 - EPS_LOOSE);
    }

    #[test]
    fn expander_quality_k6() {
        let mut g = SpectralGraph::new(6, false);
        g.build_complete(1.0);
        let q = g.expander_quality().unwrap();
        // K_6 is 5-regular, bound = 2√4 = 4.0, max_abs = 1
        // quality = 4.0 / 1.0 = 4.0
        assert!(q >= 1.0 - EPS_LOOSE);
    }

    #[test]
    fn spectral_gap_algebraic() {
        let mut g = SpectralGraph::new(4, false);
        g.build_complete(1.0);
        let gap = g.spectral_gap().unwrap();
        // K_4: eigenvalues = [-1, -1, -1, 3], gap = 3 - (-1) = 4
        assert!((gap - 4.0).abs() < EPS_LOOSE);
    }

    // ── 10. Regression Tests (Audit Bugs) ──────────────────

    #[test]
    fn regression_power_iter_zero_matrix() {
        let m = DMatrix::zeros(3, 3);
        let result = SpectralGraph::power_iteration(&m, 100, 1e-10);
        assert_eq!(result, Err(SgError::SingularMatrix));
    }

    #[test]
    fn regression_mixing_disconnected_is_none() {
        let mut g = SpectralGraph::new(6, false);
        g.add_edge(0, 1, 1.0).ok();
        g.add_edge(1, 2, 1.0).ok();
        g.add_edge(3, 4, 1.0).ok();
        g.add_edge(4, 5, 1.0).ok();
        g.finalize();
        assert!(!g.is_connected());
        assert!(g.mixing_time().is_none());
    }

    #[test]
    fn regression_conductance_weighted() {
        let mut g = SpectralGraph::new(4, false);
        g.add_edge(0, 1, 5.0).ok();
        g.add_edge(1, 2, 1.0).ok();
        g.add_edge(2, 3, 5.0).ok();
        g.add_edge(3, 0, 1.0).ok();
        g.finalize();

        let s = vec![0usize, 1];
        let cond = g.conductance_of_set(&s);
        // cut edges: (1,2) w=1.0 + (0,3) w=1.0 = 2.0
        // vol(S) = 6+6=12, vol(comp) = 6+6=12
        // cond = 2/12 = 1/6
        assert!((cond - 1.0 / 6.0).abs() < EPS_LOOSE);
    }

    #[test]
    fn regression_expander_quality_higher_for_better() {
        let mut gk = SpectralGraph::new(6, false);
        gk.build_complete(1.0);
        let qk = gk.expander_quality().unwrap();

        let mut gc = SpectralGraph::new(6, false);
        gc.build_cycle(1.0);
        let qc = gc.expander_quality().unwrap();

        assert!(qk >= qc);
    }

    #[test]
    fn regression_eigendecompose_consistency() {
        let mut m = nalgebra::DMatrix::zeros(3, 3);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 0.5;
        m[(1, 0)] = 0.5;
        m[(1, 1)] = 5.0;
        m[(1, 2)] = 0.3;
        m[(2, 1)] = 0.3;
        m[(2, 2)] = 10.0;

        let (eigenvalues, _) = SpectralGraph::eigendecompose(&m).unwrap();
        assert!((eigenvalues[0] - 1.0).abs() < 0.5);
        assert!((eigenvalues[1] - 5.0).abs() < 0.5);
        assert!((eigenvalues[2] - 10.0).abs() < 0.5);
    }

    #[test]
    fn regression_fiedler_value_not_absolute() {
        // Ensure Fiedler returns algebraic value, not absolute
        let mut g = SpectralGraph::new(4, false);
        g.build_complete(1.0);
        let f = g.fiedler_value().unwrap();
        // K_4 Laplacian eigenvalues: [0, 4, 4, 4]
        assert!(f > 0.0);
    }

    // ── 11. Robustness ─────────────────────────────────────

    #[test]
    fn robustness_complete() {
        let mut g = SpectralGraph::new(4, false);
        g.build_complete(1.0);
        let r = g.robustness();
        assert!(r > 0.0);
        assert_eq!(g.count_articulation_points(), 0);
    }

    #[test]
    fn robustness_path_has_articulation_points() {
        let mut g = SpectralGraph::new(5, false);
        g.build_path(1.0);
        assert!(g.count_articulation_points() > 0);
    }

    #[test]
    fn more_connected_more_robust() {
        let mut g1 = SpectralGraph::new(5, false);
        g1.build_complete(1.0);
        let r1 = g1.robustness();

        let mut g2 = SpectralGraph::new(5, false);
        g2.build_path(1.0);
        let r2 = g2.robustness();

        assert!(r1 > r2);
    }

    // ── 12. Spectral Clustering ────────────────────────────

    #[test]
    fn cluster_two_clusters() {
        let mut g = SpectralGraph::new(6, false);
        g.add_edge(0, 1, 1.0).ok();
        g.add_edge(0, 2, 1.0).ok();
        g.add_edge(1, 2, 1.0).ok();
        g.add_edge(3, 4, 1.0).ok();
        g.add_edge(3, 5, 1.0).ok();
        g.add_edge(4, 5, 1.0).ok();
        g.add_edge(2, 3, 1.0).ok();
        g.finalize();

        let clusters = g.spectral_cluster(2).unwrap();
        assert_eq!(clusters[0], clusters[1]);
        assert_eq!(clusters[3], clusters[4]);
        assert_ne!(clusters[0], clusters[3]);
    }

    #[test]
    fn cluster_one_cluster() {
        let mut g = SpectralGraph::new(3, false);
        g.build_complete(1.0);
        let clusters = g.spectral_cluster(1).unwrap();
        assert!(clusters.iter().all(|&c| c == 0));
    }

    #[test]
    fn cluster_bad_k() {
        let mut g = SpectralGraph::new(3, false);
        g.build_complete(1.0);
        assert!(g.spectral_cluster(0).is_err());
        assert!(g.spectral_cluster(5).is_err());
    }

    // ── 13. Integration Tests ──────────────────────────────

    #[test]
    fn fiedler_monotone_with_edges() {
        let mut g1 = SpectralGraph::new(4, false);
        g1.build_path(1.0);
        let f1 = g1.fiedler_value().unwrap();

        let mut g2 = SpectralGraph::new(4, false);
        g2.build_cycle(1.0);
        let f2 = g2.fiedler_value().unwrap();

        assert!(f2 >= f1 - EPS_LOOSE);
    }

    #[test]
    fn expander_cycle_vs_complete() {
        let mut gc = SpectralGraph::new(6, false);
        gc.build_cycle(1.0);
        let hc = gc.cheeger_constant().unwrap();

        let mut gk = SpectralGraph::new(6, false);
        gk.build_complete(1.0);
        let hk = gk.cheeger_constant().unwrap();

        assert!(hk > hc);
    }

    #[test]
    fn edge_connectivity() {
        let mut g = SpectralGraph::new(6, false);
        g.build_cycle(1.0);
        assert!((g.edge_connectivity() - 2.0).abs() < EPS);

        let mut g2 = SpectralGraph::new(4, false);
        g2.build_complete(1.0);
        assert!((g2.edge_connectivity() - 3.0).abs() < EPS);
    }

    // ── 14. Edge Cases ─────────────────────────────────────

    #[test]
    fn small_graph_two_nodes() {
        let mut g = SpectralGraph::new(2, false);
        g.add_edge(0, 1, 1.0).ok();
        g.finalize();
        let f = g.fiedler_value().unwrap();
        assert!((f - 2.0).abs() < EPS_LOOSE);
    }

    #[test]
    fn single_node_graph() {
        let mut g = SpectralGraph::new(1, false);
        g.finalize();
        assert!(g.is_connected());
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.degree(0), 0);
    }

    #[test]
    fn weighted_graph() {
        let mut g = SpectralGraph::new(3, false);
        g.add_edge(0, 1, 2.0).ok();
        g.add_edge(1, 2, 3.0).ok();
        g.add_edge(0, 2, 1.0).ok();
        g.finalize();
        let a = g.adjacency_matrix();
        assert!((a[(0, 1)] - 2.0).abs() < EPS);
        assert!((a[(1, 2)] - 3.0).abs() < EPS);
    }

    #[test]
    fn error_display() {
        assert_eq!(SgError::NullInput.to_string(), "NULL pointer");
        assert_eq!(SgError::SingularMatrix.to_string(), "singular matrix");
        assert_eq!(SgError::InvalidParam.to_string(), "invalid parameter");
    }

    #[test]
    fn fiedler_star_graph_exact() {
        let mut g = SpectralGraph::new(5, false);
        g.build_star(1.0);
        let f = g.fiedler_value().unwrap();
        assert!((f - 1.0).abs() < EPS_LOOSE);
    }
}
