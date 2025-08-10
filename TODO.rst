TODOs
=====

--------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[18], line 18
     14 init_proc_time = time.process_time()
     16 rho_pred, _ = ff.decompress(subsizes[i], x, method='newton', max_iter=500,
     17                             step_size=0.1, tolerance=1e-4)
---> 18 eig_pred = numpy.sort(fa.sample(x, rho_pred, subsize, seed=0))
     20 rhos_pred.append(numpy.copy(rho_pred))
     21 eigs_pred.append(numpy.copy(eig_pred))

File ~/programs/miniconda3/lib/python3.12/site-packages/freealg/_sample.py:117, in sample(x, rho, num_pts, method, seed)
    115     u = rng.random(num_pts)
    116 elif method == 'qmc':
--> 117     engine = qmc.Halton(d=1, rng=rng)
    118     u = engine.random(num_pts)
    119 else:

TypeError: Halton.__init__() got an unexpected keyword argument 'rng'

