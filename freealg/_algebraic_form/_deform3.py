# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE

import numpy
from ._poly_util import eval_P_partials

__all__ = ['deform_newton']


def _solve_wbar(z, tau, coeffs, c0, wbar0, max_iter=50, tol=1e-12, armijo=1e-4,
                min_lam=1e-6, w_min=1e-14):
    """Solve for companion transform wbar via Newton on implicit polynomial."""
    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    wbar = complex(wbar0)
    if abs(wbar) < w_min:
        wbar = w_min + 1j * max(w_min, z.imag)

    beta = (1.0 / float(tau)) - 1.0

    for _ in range(int(max_iter)):
        if abs(wbar) < w_min:
            wbar = w_min + 1j * max(w_min, z.imag)

        zeta = (z / tau) + (beta / wbar)
        if abs(zeta) < w_min:
            zeta = (w_min + 1j * z.imag)

        m = (wbar - (c0 - 1.0) / zeta) / c0
        P, Pz, Pm = eval_P_partials(zeta, m, coeffs)
        F = P
        if abs(F) <= tol:
            return wbar, True

        dzeta_dw = -beta / (wbar * wbar)
        dm_dw = (1.0 / c0) + ((c0 - 1.0) / c0) * (dzeta_dw / (zeta * zeta))
        dF = Pz * dzeta_dw + Pm * dm_dw
        if abs(dF) < 1e-30:
            return wbar, False

        step = -F / dF
        lam = 1.0
        F0 = abs(F)
        success = False
        while lam >= min_lam:
            w_try = wbar + lam * step
            if (z.imag > 0.0) and (w_try.imag <= 0.0):
                w_try = complex(w_try.real, max(w_min, abs(w_try.imag)))
            if abs(w_try) < w_min:
                w_try = w_min + 1j * max(w_min, z.imag)

            zeta_try = (z / tau) + (beta / w_try)
            if abs(zeta_try) < w_min:
                zeta_try = (w_min + 1j * z.imag)
            m_try = (w_try - (c0 - 1.0) / zeta_try) / c0
            P_try, _, _ = eval_P_partials(zeta_try, m_try, coeffs)
            F_try = abs(P_try)

            if F_try <= (1.0 - armijo * lam) * F0:
                wbar = w_try
                success = True
                break
            lam *= 0.5

        if not success:
            return wbar, False

        if abs(step) * lam <= tol * max(1.0, abs(wbar)):
            return wbar, True

    return wbar, False


def deform_newton(z_list, t_grid, coeffs, c0, w0_list, dt_max=0.1, sweep=True,
                  time_rel_tol=5.0, active_imag_eps=None, sweep_pad=20,
                  max_iter=50, tol=1e-12, armijo=1e-4, min_lam=1e-6,
                  w_min=1e-14):
    """Evolve m(t,z) under the Silverstein / aspect-ratio scaling flow."""

    z_list = numpy.asarray(z_list, dtype=complex).ravel()
    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    nt = t_grid.size
    nz = z_list.size
    if nt == 0:
        raise ValueError("t_grid must be non-empty.")
    if abs(t_grid[0]) > 1e-15:
        raise ValueError("t_grid must start at 0 (tau=1).")
    if numpy.any(numpy.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing.")

    w0_list = numpy.asarray(w0_list, dtype=complex).ravel()
    if w0_list.size != nz:
        raise ValueError("w0_list must have the same length as z_list.")

    wbar_prev = c0 * w0_list + (c0 - 1.0) / z_list

    W = numpy.empty((nt, nz), dtype=complex)
    ok = numpy.zeros((nt, nz), dtype=bool)
    W[0, :] = w0_list
    ok[0, :] = True
    ok_prev = numpy.ones(nz, dtype=bool)

    if active_imag_eps is None:
        active_imag_eps = 50.0 * float(abs(z_list[0].imag))

    active_hold_substeps = 2
    active_hold = numpy.zeros(nz, dtype=numpy.int64)

    def _score_candidate(wcand, okcand, iz, w_time, w_left=None, w_right=None):
        if not okcand:
            return numpy.inf
        # Prefer continuity in time and space; discourage collapsing Im part.
        score = abs(wcand - w_time)
        if w_left is not None:
            score += 0.5 * abs(wcand - w_left)
        if w_right is not None:
            score += 0.5 * abs(wcand - w_right)
        # Soft penalty for abrupt drop in imaginary part when time seed is active.
        im_prev = abs(w_time.imag)
        if im_prev > float(active_imag_eps):
            if abs(wcand.imag) < 0.2 * im_prev:
                score += 2.0 * im_prev
        return score

    def _solve_with_seed(iz, t, w_seed, w_time):
        tau = float(numpy.exp(t))
        wbar, success = _solve_wbar(
            z_list[iz], tau, coeffs, float(c0), w_seed,
            max_iter=max_iter, tol=tol, armijo=armijo,
            min_lam=min_lam, w_min=w_min)
        return wbar, success

    def _solve_at_index(iz, t, w_seed, w_time):
        # Keep original helper behavior for non-sweep path and compatibility.
        wbar, success = _solve_with_seed(iz, t, w_seed, w_time)
        if not success:
            return wbar, False
        if (time_rel_tol is not None) and (time_rel_tol > 0.0):
            if abs(wbar - w_time) > time_rel_tol * max(1.0, abs(w_time)):
                wbar2, success2 = _solve_with_seed(iz, t, w_time, w_time)
                if success2:
                    return wbar2, True
        return wbar, True

    def _solve_multi_seed(iz, t, seeds, w_time, w_left=None, w_right=None):
        # Deduplicate seeds lightly (order-preserving).
        uniq = []
        for s in seeds:
            if s is None:
                continue
            s = complex(s)
            keep = True
            for u in uniq:
                if abs(s - u) <= 1e-14 * max(1.0, abs(u)):
                    keep = False
                    break
            if keep:
                uniq.append(s)

        best_w = complex(w_time)
        best_ok = False
        best_score = numpy.inf
        for s in uniq:
            w, okc = _solve_with_seed(iz, t, s, w_time)
            sc = _score_candidate(w, okc, iz, w_time, w_left=w_left, w_right=w_right)
            if sc < best_score:
                best_score = sc
                best_w = w
                best_ok = okc
        return best_w, best_ok

    def _build_active_pad(wbar_prev_local):
        active_now = (numpy.abs(numpy.imag(wbar_prev_local)) > float(active_imag_eps))
        active_hold[active_now] = int(active_hold_substeps)
        active_hold[~active_now] = numpy.maximum(0, active_hold[~active_now] - 1)
        active = active_now | (active_hold > 0)

        pad_label = -numpy.ones(nz, dtype=numpy.int64)
        active_pad = numpy.zeros(nz, dtype=bool)
        idx = numpy.flatnonzero(active)
        if idx.size > 0:
            cuts = numpy.where(numpy.diff(idx) > 1)[0]
            blocks = numpy.split(idx, cuts + 1)
            centers = []
            pads = []
            for b in blocks:
                centers.append(int((b[0] + b[-1]) // 2))
                lo = int(max(0, b[0] - int(sweep_pad)))
                hi = int(min(nz - 1, b[-1] + int(sweep_pad)))
                pads.append((lo, hi))
            for lo, hi in pads:
                active_pad[lo:hi + 1] = True
            idx_u = numpy.flatnonzero(active_pad)
            c_cent = numpy.asarray(centers, dtype=numpy.int64)
            dist = numpy.abs(idx_u[:, None] - c_cent[None, :])
            winner = numpy.argmin(dist, axis=1).astype(numpy.int64)
            pad_label[idx_u] = winner
        return active_pad, pad_label

    def _sweep_once(t, wbar_prev_local, active_pad, pad_label, direction='lr'):
        row = numpy.empty(nz, dtype=complex)
        okr = numpy.zeros(nz, dtype=bool)
        if direction == 'lr':
            indices = range(nz)
            step = -1
        else:
            indices = range(nz - 1, -1, -1)
            step = 1

        for iz in indices:
            neigh = iz + step
            seeds = [wbar_prev_local[iz]]  # time seed always
            if 0 <= neigh < nz:
                same_comp = (active_pad[iz] and active_pad[neigh] and
                             (pad_label[iz] == pad_label[neigh]) and
                             (pad_label[iz] >= 0) and okr[neigh])
                if same_comp:
                    seeds.append(row[neigh])
            # Add nearby time seeds for robustness in weak bulks
            if iz > 0:
                seeds.append(wbar_prev_local[iz - 1])
            if iz + 1 < nz:
                seeds.append(wbar_prev_local[iz + 1])

            w_left = row[iz - 1] if (iz > 0 and okr[iz - 1]) else None
            w_right = row[iz + 1] if (iz + 1 < nz and okr[iz + 1]) else None
            w, okc = _solve_multi_seed(iz, t, seeds, wbar_prev_local[iz],
                                       w_left=w_left, w_right=w_right)
            row[iz] = w
            okr[iz] = okc
        return row, okr

    def _merge_rows(t, wbar_prev_local, row_lr, ok_lr, row_rl, ok_rl):
        row = numpy.empty(nz, dtype=complex)
        okr = numpy.zeros(nz, dtype=bool)
        for iz in range(nz):
            w_left = row[iz - 1] if (iz > 0 and okr[iz - 1]) else None
            sc_lr = _score_candidate(row_lr[iz], ok_lr[iz], iz, wbar_prev_local[iz], w_left=w_left)
            sc_rl = _score_candidate(row_rl[iz], ok_rl[iz], iz, wbar_prev_local[iz], w_left=w_left)
            if sc_lr <= sc_rl:
                row[iz] = row_lr[iz]
                okr[iz] = ok_lr[iz]
            else:
                row[iz] = row_rl[iz]
                okr[iz] = ok_rl[iz]

        # Repair unresolved points using neighbor midpoint inside resolved stretches.
        for iz in range(1, nz - 1):
            if okr[iz]:
                continue
            if okr[iz - 1] and okr[iz + 1]:
                w_seed = 0.5 * (row[iz - 1] + row[iz + 1])
                w_new, ok_new = _solve_multi_seed(iz, t, [w_seed, wbar_prev_local[iz]],
                                                  wbar_prev_local[iz],
                                                  w_left=row[iz - 1], w_right=row[iz + 1])
                if ok_new:
                    row[iz] = w_new
                    okr[iz] = True
        return row, okr

    for it in range(1, nt):
        t0 = float(t_grid[it - 1])
        t1 = float(t_grid[it])
        dt = t1 - t0
        n_sub = int(numpy.ceil(dt / float(dt_max)))
        if n_sub < 1:
            n_sub = 1

        for ks in range(1, n_sub + 1):
            t = t0 + dt * (ks / float(n_sub))

            if not sweep:
                wbar_row = numpy.empty(nz, dtype=complex)
                ok_row = numpy.zeros(nz, dtype=bool)
                for iz in range(nz):
                    wbar_row[iz], ok_row[iz] = _solve_at_index(
                        iz, t, wbar_prev[iz], wbar_prev[iz])
                wbar_prev = wbar_row
                ok_prev = ok_row
                continue

            active_pad, pad_label = _build_active_pad(wbar_prev)

            # Two directional sweeps with multi-seed solves, then merge.
            row_lr, ok_lr = _sweep_once(t, wbar_prev, active_pad, pad_label, direction='lr')
            row_rl, ok_rl = _sweep_once(t, wbar_prev, active_pad, pad_label, direction='rl')
            wbar_row, ok_row = _merge_rows(t, wbar_prev, row_lr, ok_lr, row_rl, ok_rl)

            # Keep the local rescue pass from deform2 inside active padded regions.
            for iz in range(1, nz - 1):
                if not (active_pad[iz] and (pad_label[iz] >= 0)):
                    continue
                jumpy = ok_row[iz] and (
                    abs(wbar_row[iz] - wbar_prev[iz]) >
                    2.0 * float(time_rel_tol) * max(1.0, abs(wbar_prev[iz]))
                ) if (time_rel_tol is not None and time_rel_tol > 0.0) else False
                if ok_row[iz] and (not jumpy):
                    continue

                seeds = [wbar_prev[iz]]
                if active_pad[iz - 1] and ok_row[iz - 1] and (pad_label[iz - 1] == pad_label[iz]):
                    seeds.append(wbar_row[iz - 1])
                if active_pad[iz + 1] and ok_row[iz + 1] and (pad_label[iz + 1] == pad_label[iz]):
                    seeds.append(wbar_row[iz + 1])
                if len(seeds) >= 3:
                    seeds.append(0.5 * (seeds[-1] + seeds[-2]))

                w_left = wbar_row[iz - 1] if ok_row[iz - 1] else None
                w_right = wbar_row[iz + 1] if ok_row[iz + 1] else None
                w_new, ok_new = _solve_multi_seed(iz, t, seeds, wbar_prev[iz],
                                                  w_left=w_left, w_right=w_right)
                if ok_new:
                    if (not ok_row[iz]) or (
                        _score_candidate(w_new, True, iz, wbar_prev[iz], w_left=w_left, w_right=w_right)
                        < _score_candidate(wbar_row[iz], ok_row[iz], iz, wbar_prev[iz], w_left=w_left, w_right=w_right)
                    ):
                        wbar_row[iz] = w_new
                        ok_row[iz] = True

            wbar_prev = wbar_row
            ok_prev = ok_row

        c = float(c0) * float(numpy.exp(t1))
        W[it, :] = (wbar_prev - (c - 1.0) / z_list) / c
        ok[it, :] = ok_row

    return W, ok
