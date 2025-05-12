# Install ;#\freealg#; with ;#"\texttt{pip install \freealg}"#;
import freealg as fa

# Create an object for Marchenko-Pastur distribution with ;#$ \lambda = \frac{1}{50} $#;
mp = fa.distributions.MarchenkoPastur(1/50)

# Generate a matrix corresponding to this distribution
A = mp.matrix(size=3000)

# Create a free form for the matrix within the support ;#$ I = [\lambda_{-}, \lambda_{+}] $#;
ff = fa.FreeForm(A, support=(mp.lam_m, mp.lam_p))

# Fit the distribution using Jacobi Polynomials of degree ;#$ K=20 $#;, ;#$ \alpha, \beta = 0 $#;
psi = ff.fit(method='jacobi', K=10, alpha=0.0, beta=0.0, reg=1e-2, damp='jackson', plot=True)

# Estimate empirical spectral density ;#$ \rho(x) $#;
rho = ff.density(plot=True)

# Estimate Hilbert transform ;#$ \mathcal{H}[\rho](x) $#;
hilb = ff.hilbert(plot=True)

# Estimate Stieltjes transform ;#$ m(z) $#; (both Riemannian sheets ;#$ m^{+} $#; and ;#$ m^{-}$ #;)
mp, mm = ff.stieltjes(p=1, q=1, plot=True)

# Decompress to larger matrix
rho = ff.decompress(size=100_000)