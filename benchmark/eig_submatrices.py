#! /usr/bin/env python

"""
Install requirements:

    pip3 install torch torchvision

How to use:

    * In the "main" function set the parameters in the "Setting" section.
    * Run the script with:
        python ./eig_submatrices.py
"""

# =======
# Imports
# =======

import os
import numpy
import scipy
import freealg
import time
from datetime import datetime
import torch
from pathlib import Path
import platform
import subprocess
from packaging.version import Version

if Version(freealg.__version__) < Version("0.10.0"):
    raise RuntimeError('You should use a newer version of "freealg".')


# ============
# get cpu name
# ============

def get_cpu_name():
    """
    Processor name

    Works on Linux, MacOS, and the other one!
    """

    name = platform.processor() or platform.machine() or "unknown"

    try:
        system = platform.system()

        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        name = line.split(":", 1)[1].strip()
                        break

        elif system == "Darwin":
            for key in ("machdep.cpu.brand_string", "hw.model"):
                try:
                    name = subprocess.check_output(
                        ["sysctl", "-n", key],
                        text=True
                    ).strip()
                    if name:
                        break
                except Exception:
                    pass

        elif system == "Windows":
            name = subprocess.check_output(
                ["wmic", "cpu", "get", "name"],
                text=True).splitlines()[1].strip()

    except Exception:
        pass

    return name


# ====
# load
# ====

def load(input_dir, input_filename, input_size=None, input_dtype=None,
         verbose=True):
    """
    Loads the following formats:

    * Numpy   (as *.npy)
    * memmap  (as *.dat)
    * Pytorch (as *.pt )

    Notes
    -----

    For memmap (as *dat file), both "input_size" and "dtype" must be given.
    For any other format, they are not needed.
    """

    if input_filename == '':
        return None

    # Load input matrix
    full_input_filename = os.path.join(input_dir, input_filename)
    pth = Path(full_input_filename)

    if pth.suffix == '.npy':
        if verbose:
            print(f'Loading numpy array "{input_filename}" ... ', end='',
                  flush=True)
        A = numpy.load(full_input_filename)

    elif pth.suffix == '.pt':
        if verbose:
            print(f'Loading torch array "{input_filename}" ... ', end='',
                  flush=True)
        A_t = torch.load(full_input_filename)
        A = A_t.numpy()

    elif pth.suffix == '.dat':
        if (input_size is None) or (input_dtype is None):
            raise ValueError('For memmap "dat" files, "input_size" and '
                             '"dtype" should be given.')
        if verbose:
            print(f'Loading memmap array "{input_filename}" ... ', end='',
                  flush=True)
        A_mem = numpy.memmap(full_input_filename, dtype=input_dtype, mode="r",
                             shape=(input_size, input_size))
        A = numpy.array(A_mem)

    else:
        raise ValueError('File type is not supported.')

    if verbose:
        print('done.', flush=True)

    return A


# ==============
# sampling sizes
# ==============

def sampling_sizes(n, base_factor=1000, base_exp=2, exp_incr=1,
                   num_sizes=None, max_repeat=10, verbose=True):
    """
    --------------
    Sampling sizes
    --------------

    Generate sizes n_i based on

        n_i = a * b^{t_i},   with    t_i = 0, r, 2*r, 3*r, ..., m*r

    where:

    * "a" is base_factor
    * "b" is base_exp (base exponential)
    * "r" is exp_incr (increment of exponentials)
    * "m" is num_sizes. If None, it samples to the largest possible size

    Default arguments for a 64K by 64K input matrix produces the sizes:

        1K, 2K, 4K, ..., 64K.

    -------
    Repeats
    -------

    Number of times to redo the sampling each sub matrix.

    The largest size (size of the original matrix) gets sampled once. The
    smaller the submatrix gets, more it is sampled. The smallest submatrix
    is sampled the most, which is set by max_repeat.
    """

    max_pow = int(numpy.log2(n // base_factor))
    sizes = base_factor * (base_exp ** numpy.arange(0, max_pow+1e-8, exp_incr))
    sizes = sizes.astype(int)
    if sizes[-1] < n:
        sizes = numpy.append(sizes, n)

    # Repeats
    ratio = sizes / sizes[0]
    repeats = max_repeat / ratio
    repeats = repeats.astype(int)
    repeats[repeats < 1] = 1

    if num_sizes is not None:
        sizes = sizes[:num_sizes]
        repeats = repeats[:num_sizes]

    if verbose:
        print('', flush=True)
        print('  Size  Repeat', flush=True)
        print('------  ------', flush=True)
        for i in range(sizes.size):
            print(f'{sizes[i]:>6d}  {repeats[i]:>6d}', flush=True)
        print('', flush=True)

    return sizes, repeats


# ====
# main
# ====

def main():
    """
    This function computes the eigenvalues of the submatrices of a given input
    matrix.

    -----
    Input
    -----

    The input matrix can be given in two ways:

    1. Square matrix A. This case computes the eigenvalues of submatrices of A.
    2. Tall data matrix X and 1D array d for diagonal of matrix D. This case
       computes the eigenvalues of the submatrices of A = X @ D @ X.T.

    ------
    Output
    ------

    The output is the dictionary "out" with the fields including:

    * out['sizes']: 1D array of subsampling sizes.
    * out['repeats']: 1D array of the number of times that sampling is repeated
      per each subsample size.
    * out['eigs']: list of 2D arrays. The i-th element of the list is a numpy
      array of the shape (repeats[i], sizes[i]). That is, the rows are
      eigenvalues of each repeated sampling.
    """

    # --------
    # Settings
    # --------

    # Base directory of input file(s)
    input_dir = '/data/sameli/free-algebraic'

    # --------------------------------------

    # Data matrix A (or X)
    # input_data_filename = 'ntk_cifar_resnet50_fp64_n64K.dat'
    # input_data_filename = 'ntk_cifar_resnet9_fp64_n64K.dat'
    # input_data_filename = 'hessian2100k32k.pt'
    # input_data_filename = 'hessian120k16k.pt'
    # input_data_filename = 'hessian120k16khomo.pt'
    # input_data_filename = 'Umatrix16k.pt'
    # input_data_filename = 'Umatrix64k.pt'
    input_data_filename = 'Umatrix64kf3.2n100d0.01t.npy'
    # input_data_filename = 'deformed_mp.npy'
    # input_data_filename = 'dataX120k16k.pt'

    # --------------------------------------

    # Diagonal matrix D (if given, otherwise set to empty string)
    input_diag_filename = ''
    # input_diag_filename = 'diag120k16k.pt'

    # --------------------------------------

    # These are only needed when input file is memmap (*.dat file), otherwise
    # leave them as None
    input_size = None
    # input_size = 64_000

    input_dtype = None
    # input_dtype = numpy.float64

    # ---------------------------------------

    # Whether to preserve block structure in sampling.
    block_size = None       # For all matrices except NTK
    # block_size = 10       # For NTK, set number of classes

    # ---------------------------------------

    # How to generate submatrix sizes:
    base_factor = 1000    # Smallest submatrix (such as 1K)
    base_exp = 2          # Exponential base (such as 1K, 2K, 4K, ...)
    exp_incr = 1          # Exponential increment (2^0, 2^1, 2^2, ...)
    num_sizes = None      # Number of sizes. "None" samples till full matrix

    # ---------------------------------------

    # Repeat sampling per each submatrix. The largest matrix get sampled only
    # once, smaller matrices get more sampling. The smallest submatrix gets the
    # most sampling, set by the number below.
    max_repeat = 16

    # ---------------------------------------

    verbose = True

    # ---------------------------------------

    # Load data matrix
    A = load(input_dir, input_data_filename, input_size=input_size,
             input_dtype=input_dtype, verbose=verbose)

    # Load diagonal matrix (if given, otherwise it returns None)
    d = load(input_dir, input_diag_filename, verbose=verbose)

    # n: number of data, p: number of features
    n, p = A.shape

    if n > p:
        # Tall matrix, use SVD and compute eig from their squares
        gram = True

        # If diagonals are given, replace A with A @ sqrt(diag(d))
        if d is not None:
            print('Multiplying diagonals ... ', end='', flush=True)
            d12 = numpy.sqrt(d)
            A *= d12[:, None]
            print('done.', flush=True)

    elif n == p:
        # Full square matrix is given. No need to consider it's Gram
        gram = False
    else:
        raise ValueError('"n" cannot be smaller than "p".')

    # Sampling sizes and repeat per each sampling
    sizes, repeats = sampling_sizes(n, base_factor=base_factor,
                                    base_exp=base_exp, exp_incr=exp_incr,
                                    num_sizes=num_sizes, max_repeat=max_repeat,
                                    verbose=verbose)

    eigs = []
    wall_times = []
    proc_times = []

    for i in range(sizes.size):

        eig = numpy.zeros((repeats[i], sizes[i]), dtype=numpy.float64)
        wall_time = 0.0
        proc_time = 0.0

        for j in range(repeats[i]):

            # Sample
            if verbose:
                print(f'{i+1:>2}/{sizes.size} | '
                      f'size: {sizes[i]:>5d} | '
                      f'rep {j+1:>2d}/{repeats[i]:>2d} | ',
                      end='', flush=True)
            if gram:
                # Sample rows of tall matrix (n, p) to (size, p)
                As = A[:sizes[i], :]
            else:
                # Sample (p, p) matrix to principal sub-matrix (size, size)
                if sizes[i] < A.shape[0]:
                    As = freealg.submatrix(A, sizes[i], block_size=block_size,
                                           paired=True, seed=j)
                else:
                    As = A
            if verbose:
                print('sampled | compute eig ... ', end='', flush=True)

            init_wall_time = time.time()
            init_proc_time = time.process_time()

            if gram:
                # Singular values of tall rectangular matrix
                s = scipy.linalg.svd(As, compute_uv=False, full_matrices=True)
                eig[j, :] = s**2 / n

                # Translate singular values of tall matrix to eigenvalues of
                # square matrix
                if As.shape[0] > As.shape[1]:
                    eig[j, :] = numpy.pad(eig[j, :],
                                          (0, As.shape[0] - As.shape[1]))
            else:
                # Eigenvalues of square matrix
                eig[j, :] = scipy.linalg.eigvalsh(As, lower=False, driver='ev')

            wall_time += time.time() - init_wall_time
            proc_time += time.process_time() - init_proc_time

            if verbose:
                print(f'done in {wall_time:7.2f} sec', flush=True)

        if verbose and (numpy.any(repeats > 1) or i == sizes.size-1):
            print('', flush=True)

        wall_times.append(wall_time / repeats[i])
        proc_times.append(proc_time / repeats[i])
        eigs.append(eig)

    # Dictionary to save
    out = {
        'sizes': sizes,
        'repeats': repeats,
        'eigs': eigs,
        'wall_times': wall_times,
        'proc_times': proc_times,
        'shape': (n, p),
        'proc_name': get_cpu_name(),
        'num_proc': os.cpu_count(),
        'date': datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        'input_filename': input_data_filename,
    }

    # Write to output
    output_dir = '.'
    output_filename = os.path.splitext(
            os.path.basename(input_data_filename))[0] + '_eigs.npz'
    full_output_filename = os.path.join(output_dir, output_filename)
    numpy.savez(full_output_filename, out=numpy.array(out, dtype=object))

    if verbose:
        print(f'Saved to {full_output_filename}')


# ============
# script guard
# ============

if __name__ == '__main__':
    main()
