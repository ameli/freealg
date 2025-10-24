# Install ;#\freealg#; with ;#"\texttt{pip install \freealg}"#;
import freealg as fa

# Create an object for the Marchenko-Pastur distribution with the parameter ;#$ \lambda = \frac{1}{50} $#;
mp = fa.distributions.MarchenkoPastur(1/50)

# Generate a matrix of size ;#$ n_s=1000 $#; corresponding to this distribution
A = mp.matrix(size=1000)

# Create a free-form object for the matrix within the support ;#$ I = [\lambda_{-}, \lambda_{+}] $#;
ff = fa.FreeForm(A, support=(mp.lam_m, mp.lam_p))

# Fit the distribution using Jacobi polynomials of degree ;#$ K=20 $#;, with ;#$ \alpha = \beta = \frac{1}{2} $#;
# Also fit the glue function via Pade of degree ;#$ [p/q] $#; with ;#$ p=1 $#;, ;#$ q=1 $#;.
psi = ff.fit(method='jacobi', K=20, alpha=0.5, beta=0.5, reg=0.0, damp='jackson',
              pade_p=1, pade_q=1, optimizer='ls', plot=True)

# Estimate the empirical spectral density ;#$ \rho(x) $#;, similar to ;# \Cref{fig:mp-free}(a) #;
rho = ff.density(plot=True)

# Estimate the Hilbert transform ;#$ \mathcal{H}[\rho](x) $#;
hilb = ff.hilbert(plot=True)

# Estimate the Stieltjes transform ;#$ m(z) $#; (both branches ;#$ m^{+} $#; and ;#$ m^{-} $#;), similar to ;# \Cref{fig:mp-stieltjes}(a,b) #;
m1, m2 = ff.stieltjes(plot=True)

# Decompress the spectral density corresponding to a larger matrix of size ;#$ n = 2^{5} \times n_s $#;,
# similar to ;# \Cref{fig:mp-free}(c) #;
rho_large, x = ff.decompress(size=32_000, plot=True)