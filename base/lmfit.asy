/*
  Copyright (c) 2009 Philipp Stephani

  Permission is hereby granted, free of charge, to any person
  obtaining a copy of this software and associated documentation files
  (the "Software"), to deal in the Software without restriction,
  including without limitation the rights to use, copy, modify, merge,
  publish, distribute, sublicense, and/or sell copies of the Software,
  and to permit persons to whom the Software is furnished to do so,
  subject to the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

/*
  Fitting $n$ data points $(x_1, y_1 \pm \Delta y_1), \dots, (x_n, y_n \pm \Delta y_n)$
  to a function $f$ that depends on $m$ parameters $a_1, \dots, a_m$ means minimizing
  the least-squares sum
  %
  \begin{equation*}
  \sum_{i = 1}^n \left( \frac{y_i - f(a_1, \dots, a_m; x_i)}{\Delta y_i} \right)^2
  \end{equation*}
  %
  with respect to the parameters $a_1, \dots, a_m$.
*/

/*
  This module provides an implementation of the Levenberg--Marquardt
  (LM) algorithm, converted from the C lmfit routine by Joachim Wuttke
  (see http://www.messen-und-deuten.de/lmfit/).

  Implementation strategy: Fortunately, Asymptote's syntax is very
  similar to C, and the original code cleanly separates the
  customizable parts (user-provided data, output routines, etc.) from
  the dirty number crunching.  Thus, mst of the code was just copied
  and slightly modified from the original source files.  I have
  amended the lm_data_type structure and the callback routines with a
  weight array that can be used to provide experimental errors.  I
  have also created two simple wrapper functions.
*/


// copied from the C code
private real LM_MACHEP = realEpsilon;
private real LM_DWARF = realMin;
private real LM_SQRT_DWARF = sqrt(realMin);
private real LM_SQRT_GIANT = sqrt(realMax);
private real LM_USERTOL = 30 * LM_MACHEP;

restricted string lm_infmsg[] = {
  "improper input parameters",
  "the relative error in the sum of squares is at most tol",
  "the relative error between x and the solution is at most tol",
  "both errors are at most tol",
  "fvec is orthogonal to the columns of the jacobian to machine precision",
  "number of calls to fcn has reached or exceeded maxcall*(n+1)",
  "ftol is too small: no further reduction in the sum of squares is possible",
  "xtol too small: no further improvement in approximate solution x possible",
  "gtol too small: no further improvement in approximate solution x possible",
  "not enough memory",
  "break requested within function evaluation"
};

restricted string lm_shortmsg[] = {
  "invalid input",
  "success (f)",
  "success (p)",
  "success (f,p)",
  "degenerate",
  "call limit",
  "failed (f)",
  "failed (p)",
  "failed (o)",
  "no memory",
  "user break"
};


// copied from the C code and amended with the weight (user_w) array
struct lm_data_type {
  real[] user_t;
  real[] user_y;
  real[] user_w;
  real user_func(real user_t_point, real[] par);  
};


// Asymptote has no pointer support, so we need reference wrappers for
// the int and real types
struct lm_int_type {
  int val;
  
  void operator init(int val) {
    this.val = val;
  }
};


struct lm_real_type {
  real val;
  
  void operator init(real val) {
    this.val = val;
  }
};


// copied from the C code; the lm_initialize_control function turned
// into a constructor
struct lm_control_type {
  real ftol;
  real xtol;
  real gtol;
  real epsilon;
  real stepbound;
  real fnorm;
  int maxcall;
  lm_int_type nfev;
  lm_int_type info;

  void operator init() {
    maxcall = 100;
    epsilon = LM_USERTOL;
    stepbound = 100;
    ftol = LM_USERTOL;
    xtol = LM_USERTOL;
    gtol = LM_USERTOL;
  }
};


// copied from the C code
typedef void lm_evaluate_ftype(real[] par, int m_dat, real[] fvec, lm_data_type data, lm_int_type info);
typedef void lm_print_ftype(int n_par, real[] par, int m_dat, real[] fvec, lm_data_type data, int iflag, int iter, int nfev);


// copied from the C code
private real SQR(real x) {
  return x * x;
}


// Asymptote doesn't support pointers to arbitrary array elements, so
// we provide an offset parameter.
private real lm_enorm(int n, real[] x, int offset=0) {
  real s1 = 0;
  real s2 = 0;
  real s3 = 0;
  real x1max = 0;
  real x3max = 0;
  real agiant = LM_SQRT_GIANT / n;
  real xabs, temp;

  for (int i = 0; i < n; ++i) {
    xabs = fabs(x[offset + i]);
    if (xabs > LM_SQRT_DWARF && xabs < agiant) {
      s2 += SQR(xabs);
      continue;
    }

    if (xabs > LM_SQRT_DWARF) {
      if (xabs > x1max) {
        temp = x1max / xabs;
        s1 = 1 + s1 * SQR(temp);
        x1max = xabs;
      } else {
        temp = xabs / x1max;
        s1 += SQR(temp);
      }
      continue;
    }
    if (xabs > x3max) {
      temp = x3max / xabs;
      s3 = 1 + s3 * SQR(temp);
      x3max = xabs;
    } else {
      if (xabs != 0.0) {
        temp = xabs / x3max;
        s3 += SQR(temp);
      }
    }
  }

  if (s1 != 0)
    return x1max * sqrt(s1 + (s2 / x1max) / x1max);
  if (s2 != 0) {
    if (s2 >= x3max)
      return sqrt(s2 * (1 + (x3max / s2) * (x3max * s3)));
    else
      return sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
  }

  return x3max * sqrt(s3);
}


// This function calculated the vector whose square sum is to be
// minimized.  We use a slight modification of the original code that
// includes the weight factor.  The user may provide different
// customizations.
void lm_evaluate_default(real[] par, int m_dat, real[] fvec, lm_data_type data, lm_int_type info) {
  for (int i = 0; i < m_dat; ++i) {
    fvec[i] = data.user_w[i] * (data.user_y[i] - data.user_func(data.user_t[i], par));
  }
}


// Helper functions to print padded strings and numbers (until
// Asymptote provides a real printf function)
private string pad(string str, int count, string pad=" ") {
  string res = str;
  while (length(res) < count)
    res = pad + res;
  return res;
}


private string pad(int num, int digits, string pad=" ") {
  return pad(string(num), digits, pad);
}


private string pad(real num, int digits, string pad=" ") {
  return pad(string(num), digits, pad);
}


// Similar to the C code, also prints weights
void lm_print_default(int n_par, real[] par, int m_dat, real[] fvec, lm_data_type data, int iflag, int iter, int nfev) {
  real f, y, t, w;
  int i;

  if (iflag == 2) {
    write("trying step in gradient direction");
  } else if (iflag == 1) {
    write(format("determining gradient (iteration %d)", iter));
  } else if (iflag == 0) {
    write("starting minimization");
  } else if (iflag == -1) {
    write(format("terminated after %d evaluations", nfev));
  }

  write("  par: ", none);
  for (i = 0; i < n_par; ++i) {
    write(" " + pad(par[i], 12), none);
  }
  write(" => norm: " + pad(lm_enorm(m_dat, fvec), 12));

  if (iflag == -1) {
    write("  fitting data as follows:");
    for (i = 0; i < m_dat; ++i) {
      t = data.user_t[i];
      y = data.user_y[i];
      w = data.user_w[i];
      f = data.user_func(t, par);
      write(format("    t[%2d]=", i) + pad(t, 12) + " y=" + pad(y, 12) + " w=" + pad(w, 12) + " fit=" + pad(f, 12) + " residue=" + pad(y - f, 12));
    }
  }
}


// Prints nothing
void lm_print_quiet(int n_par, real[] par, int m_dat, real[] fvec, lm_data_type data, int iflag, int iter, int nfev) {
}


// copied from the C code
private void lm_qrfac(int m, int n, real[] a, bool pivot, int[] ipvt, real[] rdiag, real[] acnorm, real[] wa) {
  int i, j, k, kmax, minmn;
  real ajnorm, sum, temp;
  static real p05 = 0.05;

  for (j = 0; j < n; ++j) {
    acnorm[j] = lm_enorm(m, a, j * m);
    rdiag[j] = acnorm[j];
    wa[j] = rdiag[j];
    if (pivot)
      ipvt[j] = j;
  }

  minmn = min(m, n);
  for (j = 0; j < minmn; ++j) {
    while (pivot) {
      kmax = j;
      for (k = j + 1; k < n; ++k)
        if (rdiag[k] > rdiag[kmax])
          kmax = k;
      if (kmax == j)
        break;

      for (i = 0; i < m; ++i) {
        temp = a[j * m + i];
        a[j * m + i] = a[kmax * m + i];
        a[kmax * m + i] = temp;
      }
      rdiag[kmax] = rdiag[j];
      wa[kmax] = wa[j];
      k = ipvt[j];
      ipvt[j] = ipvt[kmax];
      ipvt[kmax] = k;

      break;
    }

    ajnorm = lm_enorm(m - j, a, j * m + j);
    if (ajnorm == 0.0) {
      rdiag[j] = 0;
      continue;
    }

    if (a[j * m + j] < 0.0)
      ajnorm = -ajnorm;
    for (i = j; i < m; ++i)
      a[j * m + i] /= ajnorm;
    a[j * m + j] += 1;

    for (k = j + 1; k < n; ++k) {
      sum = 0;

      for (i = j; i < m; ++i)
        sum += a[j * m + i] * a[k * m + i];

      temp = sum / a[j + m * j];

      for (i = j; i < m; ++i)
        a[k * m + i] -= temp * a[j * m + i];

      if (pivot && rdiag[k] != 0.0) {
        temp = a[m * k + j] / rdiag[k];
        temp = max(0.0, 1 - SQR(temp));
        rdiag[k] *= sqrt(temp);
        temp = rdiag[k] / wa[k];
        if (p05 * SQR(temp) <= LM_MACHEP) {
          rdiag[k] = lm_enorm(m - j - 1, a, m * k + j + 1);
          wa[k] = rdiag[k];
        }
      }
    }

    rdiag[j] = -ajnorm;
  }
}


// copied from the C code
private void lm_qrsolv(int n, real[] r, int ldr, int[] ipvt, real[] diag, real[] qtb, real[] x, real[] sdiag, real[] wa) {
  static real p25 = 0.25;
  static real p5 = 0.5;

  int i, kk, j, k, nsing;
  real qtbpj, sum, temp;
  real _sin, _cos, _tan, _cot;

  for (j = 0; j < n; ++j) {
    for (i = j; i < n; ++i)
      r[j * ldr + i] = r[i * ldr + j];
    x[j] = r[j * ldr + j];
    wa[j] = qtb[j];
  }

  for (j = 0; j < n; ++j) {
    while (diag[ipvt[j]] != 0.0) {
      for (k = j; k < n; ++k)
        sdiag[k] = 0.0;
      sdiag[j] = diag[ipvt[j]];

      qtbpj = 0.;
      for (k = j; k < n; ++k) {
        if (sdiag[k] == 0.)
          continue;
        kk = k + ldr * k;
        if (fabs(r[kk]) < fabs(sdiag[k])) {
          _cot = r[kk] / sdiag[k];
          _sin = p5 / sqrt(p25 + p25 * _cot * _cot);
          _cos = _sin * _cot;
        } else {
          _tan = sdiag[k] / r[kk];
          _cos = p5 / sqrt(p25 + p25 * _tan * _tan);
          _sin = _cos * _tan;
        }

        r[kk] = _cos * r[kk] + _sin * sdiag[k];
        temp = _cos * wa[k] + _sin * qtbpj;
        qtbpj = -_sin * wa[k] + _cos * qtbpj;
        wa[k] = temp;

        for (i = k + 1; i < n; ++i) {
          temp = _cos * r[k * ldr + i] + _sin * sdiag[i];
          sdiag[i] = -_sin * r[k * ldr + i] + _cos * sdiag[i];
          r[k * ldr + i] = temp;
        }
      }
      break;
    }
    
    sdiag[j] = r[j * ldr + j];
    r[j * ldr + j] = x[j];
  }

  nsing = n;
  for (j = 0; j < n; ++j) {
    if (sdiag[j] == 0.0 && nsing == n)
      nsing = j;
    if (nsing < n)
      wa[j] = 0;
  }

  for (j = nsing - 1; j >= 0; --j) {
    sum = 0;
    for (i = j + 1; i < nsing; ++i)
      sum += r[j * ldr + i] * wa[i];
    wa[j] = (wa[j] - sum) / sdiag[j];
  }

  for (j = 0; j < n; ++j)
    x[ipvt[j]] = wa[j];
}


// copied from the C code
private void lm_lmpar(int n, real[] r, int ldr, int[] ipvt, real[] diag, real[] qtb, real delta, lm_real_type par, real[] x, real[] sdiag, real[] wa1, real[] wa2) {
  static real p1 = 0.1;
  static real p001 = 0.001;

  int nsing = n;
  real parl = 0.0;

  int i, iter, j;
  real dxnorm, fp, fp_old, gnorm, parc, paru;
  real sum, temp;

  for (j = 0; j < n; ++j) {
    wa1[j] = qtb[j];
    if (r[j * ldr + j] == 0 && nsing == n)
      nsing = j;
    if (nsing < n)
      wa1[j] = 0;
  }
  for (j = nsing - 1; j >= 0; --j) {
    wa1[j] = wa1[j] / r[j + ldr * j];
    temp = wa1[j];
    for (i = 0; i < j; ++i)
      wa1[i] -= r[j * ldr + i] * temp;
  }

  for (j = 0; j < n; ++j)
    x[ipvt[j]] = wa1[j];

  iter = 0;
  for (j = 0; j < n; ++j)
    wa2[j] = diag[j] * x[j];
  dxnorm = lm_enorm(n, wa2);
  fp = dxnorm - delta;
  if (fp <= p1 * delta) {
    par.val = 0;
    return;
  }

  if (nsing >= n) {
    for (j = 0; j < n; ++j)
      wa1[j] = diag[ipvt[j]] * wa2[ipvt[j]] / dxnorm;

    for (j = 0; j < n; ++j) {
      sum = 0.0;
      for (i = 0; i < j; ++i)
        sum += r[j * ldr + i] * wa1[i];
      wa1[j] = (wa1[j] - sum) / r[j + ldr * j];
    }
    temp = lm_enorm(n, wa1);
    parl = fp / delta / temp / temp;
  }

  for (j = 0; j < n; ++j) {
    sum = 0;
    for (i = 0; i <= j; ++i)
      sum += r[j * ldr + i] * qtb[i];
    wa1[j] = sum / diag[ipvt[j]];
  }
  gnorm = lm_enorm(n, wa1);
  paru = gnorm / delta;
  if (paru == 0.0)
    paru = LM_DWARF / min(delta, p1);

  par.val = max(par.val, parl);
  par.val = min(par.val, paru);
  if (par.val == 0.0)
    par.val = gnorm / dxnorm;

  for (;; ++iter) {
    if (par.val == 0.0)
      par.val = max(LM_DWARF, p001 * paru);
    temp = sqrt(par.val);
    for (j = 0; j < n; ++j)
      wa1[j] = temp * diag[j];
    lm_qrsolv(n, r, ldr, ipvt, wa1, qtb, x, sdiag, wa2);
    for (j = 0; j < n; ++j)
      wa2[j] = diag[j] * x[j];
    dxnorm = lm_enorm(n, wa2);
    fp_old = fp;
    fp = dxnorm - delta;
        
    if (fabs(fp) <= p1 * delta || (parl == 0.0 && fp <= fp_old && fp_old < 0.0) || iter == 10)
      break;
        
    for (j = 0; j < n; ++j)
      wa1[j] = diag[ipvt[j]] * wa2[ipvt[j]] / dxnorm;

    for (j = 0; j < n; ++j) {
      wa1[j] = wa1[j] / sdiag[j];
      for (i = j + 1; i < n; ++i)
        wa1[i] -= r[j * ldr + i] * wa1[j];
    }
    temp = lm_enorm(n, wa1);
    parc = fp / delta / temp / temp;
    
    if (fp > 0)
      parl = max(parl, par.val);
    else if (fp < 0)
      paru = min(paru, par.val);
    
    par.val = max(parl, par.val + parc);
  }
}


// copied from the C code; the main function
void lm_lmdif(int m, int n, real[] x, real[] fvec, real ftol, real xtol, real gtol, int maxfev, real epsfcn, real[] diag, int mode, real factor, lm_int_type info, lm_int_type nfev, real[] fjac, int[] ipvt, real[] qtf, real[] wa1, real[] wa2, real[] wa3, real[] wa4, lm_evaluate_ftype evaluate, lm_print_ftype printout, lm_data_type data) {
  static real p1 = 0.1;
  static real p5 = 0.5;
  static real p25 = 0.25;
  static real p75 = 0.75;
  static real p0001 = 1.0e-4;
  
  nfev.val = 0;
  int iter = 1;
  lm_real_type par = lm_real_type(0);
  real delta = 0;
  real xnorm = 0;
  real temp = max(epsfcn, LM_MACHEP);
  real eps = sqrt(temp);
  int i, j;
  real actred, dirder, fnorm, fnorm1, gnorm, pnorm, prered, ratio, step, sum, temp1, temp2, temp3;

  if ((n <= 0) || (m < n) || (ftol < 0.0) || (xtol < 0.0) || (gtol < 0.0) || (maxfev <= 0) || (factor <= 0)) {
    info.val = 0;
    return;
  }
  if (mode == 2) {
    for (j = 0; j < n; ++j) {
      if (diag[j] <= 0.0) {
        info.val = 0;
        return;
      }
    }
  }
  
  info.val = 0;
  evaluate(x, m, fvec, data, info);
  if(printout != null) printout(n, x, m, fvec, data, 0, 0, ++nfev.val);
  if (info.val < 0)
    return;
  fnorm = lm_enorm(m, fvec);

  do {
    for (j = 0; j < n; ++j) {
      temp = x[j];
      step = eps * fabs(temp);
      if (step == 0.0)
        step = eps;
      x[j] = temp + step;
      info.val = 0;
      evaluate(x, m, wa4, data, info);
      if(printout != null) printout(n, x, m, wa4, data, 1, iter, ++nfev.val);
      if (info.val < 0)
        return;
      for (i = 0; i < m; ++i)
        fjac[j * m + i] = (wa4[i] - fvec[i]) / (x[j] - temp);
      x[j] = temp;
    }
    
    lm_qrfac(m, n, fjac, true, ipvt, wa1, wa2, wa3);

    if (iter == 1) {
      if (mode != 2) {
        for (j = 0; j < n; ++j) {
          diag[j] = wa2[j];
          if (wa2[j] == 0.0)
            diag[j] = 1.0;
        }
      }
      for (j = 0; j < n; ++j)
        wa3[j] = diag[j] * x[j];
      xnorm = lm_enorm(n, wa3);
      delta = factor * xnorm;
      if (delta == 0.0)
        delta = factor;
    }

    for (i = 0; i < m; ++i)
      wa4[i] = fvec[i];

    for (j = 0; j < n; ++j) {
      temp3 = fjac[j * m + j];
      if (temp3 != 0.0) {
        sum = 0;
        for (i = j; i < m; ++i)
          sum += fjac[j * m + i] * wa4[i];
        temp = -sum / temp3;
        for (i = j; i < m; ++i)
          wa4[i] += fjac[j * m + i] * temp;
      }
      fjac[j * m + j] = wa1[j];
      qtf[j] = wa4[j];
    }

    gnorm = 0;
    if (fnorm != 0) {
      for (j = 0; j < n; ++j) {
        if (wa2[ipvt[j]] == 0) continue;
        sum = 0.0;
        for (i = 0; i <= j; ++i)
          sum += fjac[j * m + i] * qtf[i] / fnorm;
        gnorm = max(gnorm, fabs(sum / wa2[ipvt[j]]));
      }
    }

    if (gnorm <= gtol) {
      info.val = 4;
      return;
    }

    if (mode != 2) {
      for (j = 0; j < n; ++j)
        diag[j] = max(diag[j], wa2[j]);
    }

    do {
      lm_lmpar(n, fjac, m, ipvt, diag, qtf, delta, par, wa1, wa2, wa3, wa4);

      for (j = 0; j < n; ++j) {
        wa1[j] = -wa1[j];
        wa2[j] = x[j] + wa1[j];
        wa3[j] = diag[j] * wa1[j];
      }
      pnorm = lm_enorm(n, wa3);

      if (nfev.val <= 1 + n)
        delta = min(delta, pnorm);

      info.val = 0;
      evaluate(wa2, m, wa4, data, info);
      if(printout != null) printout(n, x, m, wa4, data, 2, iter, ++nfev.val);
      if (info.val < 0)
        return;

      fnorm1 = lm_enorm(m, wa4);

      if (p1 * fnorm1 < fnorm)
        actred = 1 - SQR(fnorm1 / fnorm);
      else
        actred = -1;

      for (j = 0; j < n; ++j) {
        wa3[j] = 0;
        for (i = 0; i <= j; ++i)
          wa3[i] += fjac[j * m + i] * wa1[ipvt[j]];
      }
      temp1 = lm_enorm(n, wa3) / fnorm;
      temp2 = sqrt(par.val) * pnorm / fnorm;
      prered = SQR(temp1) + 2 * SQR(temp2);
      dirder = -(SQR(temp1) + SQR(temp2));

      ratio = prered != 0 ? actred / prered : 0;

      if (ratio <= p25) {
        if (actred >= 0.0)
          temp = p5;
        else
          temp = p5 * dirder / (dirder + p5 * actred);
        if (p1 * fnorm1 >= fnorm || temp < p1)
          temp = p1;
        delta = temp * min(delta, pnorm / p1);
        par.val /= temp;
      } else if (par.val == 0.0 || ratio >= p75) {
        delta = pnorm / p5;
        par.val *= p5;
      }
      
      if (ratio >= p0001) {
        for (j = 0; j < n; ++j) {
          x[j] = wa2[j];
          wa2[j] = diag[j] * x[j];
        }
        for (i = 0; i < m; ++i)
          fvec[i] = wa4[i];
        xnorm = lm_enorm(n, wa2);
        fnorm = fnorm1;
        ++iter;
      }

      info.val = 0;
      if (fabs(actred) <= ftol && prered <= ftol && p5 * ratio <= 1)
        info.val = 1;
      if (delta <= xtol * xnorm)
        info.val += 2;
      if (info.val != 0)
        return;

      if (nfev.val >= maxfev)
        info.val = 5;
      if (fabs(actred) <= LM_MACHEP && prered <= LM_MACHEP && p5 * ratio <= 1)
        info.val = 6;
      if (delta <= LM_MACHEP * xnorm)
        info.val = 7;
      if (gnorm <= LM_MACHEP)
        info.val = 8;
      if (info.val != 0)
        return;
    } while (ratio < p0001);
  } while (true);
}


// copied from the C code; wrapper of lm_lmdif
void lm_minimize(int m_dat, int n_par, real[] par, lm_evaluate_ftype evaluate, lm_print_ftype printout, lm_data_type data, lm_control_type control) {
  int n = n_par;
  int m = m_dat;
  
  real[] fvec = new real[m];
  real[] diag = new real[n];
  real[] qtf = new real[n];
  real[] fjac = new real[n * m];
  real[] wa1 = new real[n];
  real[] wa2 = new real[n];
  real[] wa3 = new real[n];
  real[] wa4 = new real[m];
  int[] ipvt = new int[n];

  control.info.val = 0;
  control.nfev.val = 0;

  lm_lmdif(m, n, par, fvec, control.ftol, control.xtol, control.gtol, control.maxcall * (n + 1), control.epsilon, diag, 1, control.stepbound, control.info, control.nfev, fjac, ipvt, qtf, wa1, wa2, wa3, wa4, evaluate, printout, data);

  if(printout != null) printout(n, par, m, fvec, data, -1, 0, control.nfev.val);
  control.fnorm = lm_enorm(m, fvec);
  if (control.info.val < 0)
    control.info.val = 10;
}


// convenience functions; wrappers of lm_minimize

/*
  The structure FitControl specifies various control parameters.
*/
struct FitControl {
  real squareSumTolerance;      // relative error desired in the sum of squares
  real approximationTolerance;  // relative error between last two approximations
  real desiredOrthogonality;    // orthogonality desired between the residue vector and its derivatives
  real epsilon;                 // step used to calculate the jacobian
  real stepBound;               // initial bound to steps in the outer loop
  int maxIterations;            // maximum number of iterations
  bool verbose;                 // whether to print detailed information about every iteration, or nothing

  void operator init(real squareSumTolerance=LM_USERTOL, real approximationTolerance=LM_USERTOL, real desiredOrthogonality=LM_USERTOL, real epsilon=LM_USERTOL, real stepBound=100, int maxIterations=100, bool verbose=false) {
    this.squareSumTolerance = squareSumTolerance;
    this.approximationTolerance = approximationTolerance;
    this.desiredOrthogonality = desiredOrthogonality;
    this.epsilon = epsilon;
    this.stepBound = stepBound;
    this.maxIterations = maxIterations;
    this.verbose = verbose;
  }

  FitControl copy() {
    FitControl result = new FitControl;
    result.squareSumTolerance = this.squareSumTolerance;
    result.approximationTolerance = this.approximationTolerance;
    result.desiredOrthogonality = this.desiredOrthogonality;
    result.epsilon = this.epsilon;
    result.stepBound = this.stepBound;
    result.maxIterations = this.maxIterations;
    result.verbose = this.verbose;
    return result;
  }
};

FitControl operator init() {
  return FitControl();
}

FitControl defaultControl;


/*
  Upon returning, this structure provides information about the fit.
*/
struct FitResult {
  real norm;        // norm of the residue vector
  int iterations;   // actual number of iterations
  int status;       // status of minimization

  void operator init(real norm, int iterations, int status) {
    this.norm = norm;
    this.iterations = iterations;
    this.status = status;
  }
};


/*
  Fits data points to a function that depends on some parameters.

  Parameters:
  - xdata: Array of x values.
  - ydata: Array of y values.
  - errors: Array of experimental errors; each element must be strictly positive
  - function: Fit function.
  - parameters: Parameter array. Before calling fit(), this must contain the initial guesses for the parameters.
  Upon return, it will contain the solution parameters.
  - control: object of type FitControl that controls various aspects of the fitting procedure.

  Returns:
  An object of type FitResult that conveys information about the fitting process.
*/
FitResult fit(real[] xdata, real[] ydata, real[] errors, real function(real[], real), real[] parameters, FitControl control=defaultControl) {
  int m_dat = min(xdata.length, ydata.length);
  int n_par = parameters.length;
  lm_evaluate_ftype evaluate = lm_evaluate_default;
  lm_print_ftype printout = control.verbose ? lm_print_default : lm_print_quiet;
  
  lm_data_type data;
  data.user_t = xdata;
  data.user_y = ydata;
  data.user_w = 1 / errors;
  data.user_func = new real(real x, real[] params) {
    return function(params, x);
  };

  lm_control_type ctrl;
  ctrl.ftol = control.squareSumTolerance;
  ctrl.xtol = control.approximationTolerance;
  ctrl.gtol = control.desiredOrthogonality;
  ctrl.epsilon = control.epsilon;
  ctrl.stepbound = control.stepBound;
  ctrl.maxcall = control.maxIterations;

  lm_minimize(m_dat, n_par, parameters, evaluate, printout, data, ctrl);
  
  return FitResult(ctrl.fnorm, ctrl.nfev.val, ctrl.info.val);
}


/*
  Fits data points to a function that depends on some parameters.

  Parameters:
  - xdata: Array of x values.
  - ydata: Array of y values.
  - function: Fit function.
  - parameters: Parameter array. Before calling fit(), this must contain the initial guesses for the parameters.
  Upon return, it will contain the solution parameters.
  - control: object of type FitControl that controls various aspects of the fitting procedure.

  Returns:
  An object of type FitResult that conveys information about the fitting process.
*/
FitResult fit(real[] xdata, real[] ydata, real function(real[], real), real[] parameters, FitControl control=defaultControl) {
  return fit(xdata, ydata, array(min(xdata.length, ydata.length), 1.0), function, parameters, control);
}

