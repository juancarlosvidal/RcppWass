// WassersteinRegression.h: Rcpp/Wass glue
//
// Copyright (C) 2019 - 2020  Juan Carlos Vidal and Marcos Matabuena 
//
// This file is part of RcppWass.
//
// RcppWass is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RcppWass is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RcppWass.  If not, see <http://www.gnu.org/licenses/>.

#ifndef _WASSERSTEIN_REGRESSION_H // include guard
#define _WASSERSTEIN_REGRESSION_H

#include <stdlib.h>
#include <math.h>
#include <RcppArmadillo.h>


namespace wass {

#ifndef uint
typedef unsigned int uint;
#endif

struct regression_struct {
  arma::mat xfit;
  arma::mat xpred;
  arma::mat Qfit;
  arma::mat Qpred;
  arma::mat qfit;
  arma::mat qpred;
  arma::mat ffit;
  arma::mat fpred;
  int QP_used;
};

/** 
 * Cumulative numerical integration of points Y using the trapezoidal method.
 * Inputs: 
 *   X - matrix of coordinates or scalar spacing
 *   Y - matrix of points
 * Outputs: 
 *   C - matrix with the approximate cumulative integral of Y with respect to the scalar spacing specified by X.
 */
inline arma::mat cumtrapz(arma::mat X, arma::mat Y) {
  if (X.n_rows != Y.n_rows || X.n_cols != Y.n_cols)
    throw std::invalid_argument("Arguments 'x' and 'y' must be matrices of the same dimension");
  
  bool transposed = false;
  if (X.n_rows == 1) {
    // row vector
    X = X.t();
    Y = Y.t();
    transposed = true; 
  }
  
  uint n = X.n_rows;
  uint m = X.n_cols;
  arma::mat C = arma::zeros(n,m);
  arma::mat tmp = (X.rows(1,n-1) - X.rows(0,n-2)) % (Y.rows(1,n-1) + Y.rows(0,n-2));
  
  if (tmp.n_rows == 1)
    C.row(1) = 0.5 * tmp;
  else
    C.rows(1,n-1) = 0.5 * cumsum(tmp);
  
  if (transposed) 
    return C.t();
  else 
    return C;
}

inline bool is_smaller(arma::mat A, uint i, uint j) {
  for (uint k=0; k < A.n_cols; k++) {
    if (A(i,k) > A(j,k))
      return true;
    else if (A(i,k) < A(j,k))
      return false;
  } 
  return false;
}

inline uint find_index(arma::mat C, arma::rowvec rowA) {
  for (uint i=0; i < C.n_rows; i++) {
    arma::rowvec rowC = C.row(i);
    if (arma::approx_equal(rowA, rowC, "absdiff", 0.002)) 
      return i;
  }
  return 0;
}


/** 
 * This is an auxiliary function that returns a tuple with a matrix M containing the 
 * unique rows of A and a vector of indices ic such that A = M(ic). 
 * Inputs: 
 *   A - a matrix 
 * Outputs:
 *   B  - a matrix with the unique rows
 *   ic - vector of indices
 */
inline std::tuple<arma::mat, arma::uvec> ic_unique_rows(arma::mat A) {
  arma::mat B(A.n_rows, A.n_cols);
  uint count = 0;
  for (uint i=0; i < A.n_rows; i++) {
    arma::rowvec rowA = A.row(i);
    bool found = false;
    for (uint j=0; j < count; j++) {
      arma::rowvec rowB = B.row(j);
      if (arma::approx_equal(rowA, rowB, "absdiff", 0.002)) {
        // std::cout << "A row " << i << " equals B row " << j << std::endl;
        found = true;
        break;
      }
    }
    if (!found) 
      B.row(count++) = rowA;
  }
  
  B = B.rows(0, count-1);
  arma::uvec indices = sort_index(B.col(0));
  // BEGIN NEW
  for (uint i=0; i < indices.n_elem-1; i++) {
    for (uint j=i+1; j < indices.n_elem; j++) { 
      if (B(indices(i),0) < B(indices(j),0))
        break;
      // std::cout << "is smaller: " << B.row(indices(i)) << " vs " << B.row(indices(j)) << ": " << is_smaller(B,indices(i),indices(j)) << std::endl;
      if (is_smaller(B,indices(i),indices(j))) {
        uint aux = indices(i);
        indices(i) = indices(j);
        indices(j) = aux;
      }
    }
  }
  // END NEW
  arma::mat C = arma::mat(B.n_rows, B.n_cols, arma::fill::zeros);
  for (uint i=0; i < B.n_rows; i++) 
    C.row(i) = B.row(indices(i));
  
  arma::uvec ic(A.n_rows);
  for (uint i=0; i < A.n_rows; i++) {
    arma::rowvec rowA = A.row(i);
    ic(i) = find_index(C, rowA);
  }
  // arma::uvec ic(A.n_rows * A.n_cols);
  // for (uint i=0; i < A.n_rows; i++) {
  //   arma::rowvec rowA = A.row(i);
  //   ic(i) = find_index(C, rowA);
  // }
  
  // for (uint i=0; i < A.n_rows; i++) {
  //   arma::rowvec rowA = A.row(i);
  //   for (uint j=0; j < B.n_rows; j++) {
  //     arma::rowvec rowB = B.row(j);
  //     if (approx_equal(rowA, rowB, "absdiff", 0.002)) {
  //       ic(i) = j;
  //       break;
  //     }
  //   }
  // }
  
  // std::cout << "C: \n" << B << std::endl;
  // std::cout << "ic: \n" << ic << std::endl;
  
  return std::make_tuple(C, ic);
}


/** 
 * This is an auxiliary function that computes the matrix C and vector c for the quadratic program.  
 * Inputs: 
 *   t - grid vector on [0,1]
 * Outputs: 
 *   c - vector for the quadratic program
 *   C - matrix for the quadratic program
 */
inline std::tuple<arma::vec, arma::mat> getcC(arma::vec t) {
  arma::vec deltaT = arma::diff(t);
  arma::vec deltaTp = arma::join_vert(deltaT, arma::zeros(1));
  arma::vec deltaTm = arma::join_vert(arma::zeros(1), deltaT);
  
  arma::vec bm = 0.5 * (deltaTp + deltaTm);
  arma::mat Dm = bm * bm.t();
  
  uint m = deltaT.n_elem-1;
  arma::vec c = 0.5 * deltaT(m) * bm;
  arma::mat C = 0.1 * deltaT(m) * Dm;
  
  for (uint k = 0; k < m; k++) {
    arma::vec bk = 0.5 * arma::join_vert(deltaTm.rows(0, k+1) + deltaTp.rows(0, k+1), arma::zeros(m-k, 1));
    c = c + 0.5 * (deltaT(k) + deltaT(k+1)) * bk;
    arma::mat Dk = bk * bk.t();
    C = C + 0.5 * (deltaT(k) + deltaT(k+1)) * Dk;
  }
  
  return std::make_tuple(c,C);
}



/** 
 * This function perform Frechet regression with the Wasserstein distance
 * Inputs:
 *   xfit - nxp matrix of predictor values for fitting (do not include a column for the intercept)
 *   q - nxm matrix of quantile density functions. q(i, :) is a 1xm vector of quantile density function values on an equispaced grid on [0, 1]
 *	 Q0 - 1xn array of quantile function values at 0
 *   xpred - kxp matrix of input values for regressors for prediction.
 *   t - 1xm vector - common grid for all quantile density functions in q.  If missing, defaults to linspace(0, 1, m);  For best results, should use a finer grid than for quantle estimation, especially near the boundaries
 *   qdmin - a positive lower bound on the estimated quantile densites.  Defaults to 1e-6.
 * Outputs:
 *   A structure with the following fields:
 *	   xpred - see input of same name
 *	   qpred - kxN array.  qpred(l,:) is the regression prediction of q (the quantile density) given X = xpred(l, :)'
 *     Qpred - kxm array. Qpred(l, :) is the regression prediction of Q given X = xpred(l, :)'
 *	   fpred - kxm array. fpred(l, :) is the regression prediction of f (the density) given X = xpred(l, :)', evaluated on the grid Qpred(l, :)
 *	   xfit - see input of same name
 *	   qfit - nxN array. qfit(l, :) is the regression prediction of q given X = xfit(l, :)'
 *     Qfit - nxm array. Qfit(l, :) is the regression prediction of Q given X = xfit(l, :)'
 *	   fpred - kxm array. fpred(l, :) is the regression prediction of f (the density) given X = xpred(l, :)', evaluated on the grid Qfit(l, :)
 *	   QP_used - flag indicating whether OLS fits all satisfied the constraints (=0) or if the quadratic program was used in fitting (=1)
 */
inline regression_struct wasser(const arma::mat xfit, const arma::mat q, const arma::mat Q0, 
                         const arma::mat xpred, const arma::vec t, const double qdmin) {
  uint n = q.n_rows;
  uint m = q.n_cols;
  // std::cout << "n: " << n << std::endl;
  // std::cout << "m: " << m << std::endl;
  
  if (t.n_elem != m) 
    throw std::invalid_argument("Length of t should match number of columns in q");
  
  if (t(0) != 0 || t(t.n_elem-1) != 1)
    throw std::invalid_argument("Input t should be an increasing grid beginning at 0 and ending at 1");
  
  // std::cout << "xfit: " << xfit.n_rows << ", " << xfit.n_cols << std::endl;
  // std::cout << "xfit: \n" << xfit.row(0) << std::endl;
  
  
  uint k = xpred.n_rows;
  // std::cout << "k: " << k << std::endl;
  // uint p = xpred.n_cols;
  
  arma::mat sig = arma::cov(xfit);
  // std::cout << "sig: \n" << sig.row(0) << std::endl;
  
  arma::mat xbar = arma::mean(xfit, 0);
  // std::cout << "xbar: \n" << xbar.row(0) << std::endl;
  
  // arma::mat xall = unique_rows(tmp);
  arma::mat xall; 
  arma::uvec ic;
  std::tie(xall,ic) = ic_unique_rows(arma::join_vert(arma::join_vert(xpred, xfit), xbar));
  // std::cout << "xall: " << xall.n_rows << ", " << xall.n_cols << std::endl;
  // std::cout << "xall: \n" << xall.rows(0,50) << std::endl;
  // std::cout << "xall: \n" << xall << std::endl;
  // std::cout << "ic: \n" << ic(arma::span(0,10)) << std::endl;
  // std::cout << "ic: \n" << ic << std::endl;
  // arma::mat AA = arma::join_vert(arma::join_vert(xpred, xfit), xbar);
  // for (int kk=0; kk < 10; kk++) {
  //   std::cout << "A(" << kk << ",:) = " << AA.row(kk) << "C(" << ic(kk) << ",:) = " << xall.row(ic(kk)) << std::endl;
  // }
  
  uint r = xall.n_rows;
  // std::cout << "r: " << r << std::endl;
  
  
  // Get OLS fit
  arma::mat A = arma::join_horiz(arma::ones<arma::mat>(n,1), xfit);
  arma::mat B = Q0;
  // arma::vec ahat = wass::linear_solver(A.t() * A, A.t() * B);
  arma::vec ahat = arma::solve(A.t() * A, A.t() * Q0);
  // arma::mat bhat = wass::linear_solver(A.t() * A, A.t() * B);
  arma::mat bhat  = arma::solve(A.t() * A, A.t() * q);
  arma::mat qall  = arma::join_horiz(arma::ones<arma::mat>(r,1), xall) * bhat;
  arma::vec Q0all = arma::join_horiz(arma::ones<arma::mat>(r,1), xall) * ahat;
  // std::cout << "ahat: \n"  << ahat << std::endl;
  // std::cout << "bhat: \n"  << bhat << std::endl;
  // std::cout << "qall: \n"  << qall << std::endl;
  // std::cout << "Q0all: \n" << Q0all << std::endl;
  // std::cout << "bhat: \n" << bhat.row(0) << std::endl;
  // std::cout << "qall: \n" << qall.row(9) << std::endl;
  // std::cout << "Q0all: \n" << Q0all.row(0) << std::endl;
  
  // Check for positivity - if violated, project onto positive functions using quadratic program
  int QP_used = 0;
  arma::uvec dec = arma::find(arma::min(qall.t(), 0) < 0);
  // std::cout << "dec: " << dec << std::endl;
  
  if (!dec.is_empty()) {
    QP_used = 1; // Set to 1 if quadratic program was used
    
    arma::vec c; 
    arma::mat C;
    std::tie(c,C) = getcC(t);
    // std::cout << "c:" << c.n_rows << ", " << c.n_cols << std::endl;
    // std::cout << "c:\n" << c(arma::span(0,3)) << std::endl;
    // std::cout << "C:" << C.n_rows << ", " << C.n_cols << std::endl;
    // std::cout << "C:\n" << C(199,arma::span(0,3)) << std::endl;
    
    
    arma::mat D = arma::join_vert(arma::join_vert(arma::ones(1,1), c).t(), arma::join_horiz(c, C));
    // std::cout << "D: \n" << D.row(0) << std::endl;
    arma::mat V1 = arma::join_horiz(arma::zeros(m,1), - arma::eye(m,m));
    // std::cout << "V1: \n" << V1.row(0) << std::endl;
    arma::mat Qdmin = {qdmin};
    // std::cout << "Qdmin: \n" << Qdmin.row(0) << std::endl;
    arma::mat v1 = -repmat(Qdmin, 1, m);
    // std::cout << "v1: \n" << v1 << std::endl;
    arma::mat V2 = arma::join_horiz(arma::zeros(m-1, 1), 
                                    arma::join_horiz(arma::eye(m-1, m-1), 
                                                     arma::zeros(m-1, 1)) - arma::join_horiz(arma::zeros(m-1, 1), 
                                                     arma::eye(m-1, m-1)));
    // std::cout << "V2: \n" << V2.row(0) << std::endl;
    
    for (uint j=0; j < dec.n_elem; j++) {
      double ax = Q0all(dec(j));
      // std::cout << "ax: \n" << ax << std::endl;
      // std::cout << "hx: \n" << qall.row(dec(j)) << std::endl;
      arma::rowvec hx = qall.row(dec(j));
      // std::cout << "hx: \n" << hx(arma::span(0,5)) << std::endl;
      arma::vec d = - arma::join_vert(ax + c.t()*hx.t(), ax*c + C*hx.t());
      // std::cout << "d: \n" << d << std::endl;
      // This penalty induces smoothness into the quantile density estimates.
      // The multiplier of 1.5 is arbitrary, and should probably be chosen more carefully.
      arma::rowvec v2 = 1.5*abs(arma::diff(hx));
      // std::cout << "v2: \n" << v2 << std::endl;
      arma::mat V = arma::join_vert(V1, arma::join_vert(V2, -V2));
      // std::cout << "V: \n" << V << std::endl;
      arma::rowvec v = arma::join_horiz(v1, arma::join_horiz(v2, v2));
      // std::cout << "v: \n" << v << std::endl;
      
      // std::cout << "D(" << D.n_rows << "," << D.n_cols << ")" << std::endl;
      // std::cout << "d(" << d.n_rows << "," << d.n_cols << ")" << std::endl;
      // std::cout << "V(" << V.n_rows << "," << V.n_cols << ")" << std::endl;
      // std::cout << "v(" << v.n_rows << "," << v.n_cols << ")" << std::endl;
      
      
      // std::cout << "pre quadratic" << std::endl;
      arma::vec tmp(d.n_elem, arma::fill::zeros);
      try {
        // std::cout << "D:\n" << D << std::endl;
        // std::cout << "D: " << D.n_rows << ", " << D.n_cols << std::endl;
        // std::cout << "d:\n" << d << std::endl;
        // std::cout << "d: " << d.n_elem << std::endl;
        // std::cout << "V:\n" << V << std::endl;
        // std::cout << "V: " << V.n_rows << ", " << V.n_cols << std::endl;
        // std::cout << "v:\n" << v.as_col() << std::endl;
        // std::cout << "v: " << v.n_elem << std::endl;
        tmp = quadprog(D, d, V, v.t());
      } catch (...) {
        std::cout << "WARNING: An error has occurred during the quadratic optimization..." << std::endl;
      }
      // arma::vec tmp = quadprog(D, d, V,v.as_col(), lb, ub);
      // arma::vec tmp = quadprog(DENSE_AUL, D, d, V,v.as_col());
      // std::cout << "quadratic: \n" << tmp << std::endl;
      // std::cout << "post quadratic: " << tmp.n_elem << std::endl;
      qall.row(dec(j)) = tmp.subvec(1,tmp.n_elem-1).t();
      Q0all(dec(j)) = tmp(0);
      
      // std::cout << "j: " << j << " -- dec(j): " << dec(j) << std::endl;
      // if (j==0 || dec(j)==181) {
      // std::cout << "D(" << D.n_rows << "," << D.n_cols << ")" << std::endl;
      // std::cout << "d(" << d.n_rows << "," << d.n_cols << ")" << std::endl;
      // std::cout << "V(" << V.n_rows << "," << V.n_cols << ")" << std::endl;
      // std::cout << "v(" << v.n_rows << "," << v.n_cols << ")" << std::endl;
      // std::cout << "ax: " << ax << std::endl;
      // std::cout << "quadratic: \n" << tmp(arma::span(0,50)) << std::endl;
      //   std::cout << "hx: \n" << hx(arma::span(0,50)) << std::endl;
      //   std::cout << "D:\n" << D(0,arma::span(0,50)) << std::endl;
      //   std::cout << "d:\n" << d(arma::span(0,50)) << std::endl;
      //   std::cout << "V:\n" << V(0,arma::span(0,50)) << std::endl;
      //   std::cout << "v:\n" << v(arma::span(0,50)) << std::endl;
      //   
      // }
    }
  }
  // std::cout << "t: \n"  << t.t() << std::endl;
  // arma::rowvec s = arma::sum(qall);
  // for (int ii=0; ii < s.n_elem; ii++) {
  //   std::cout << "s(" << ii << ")= " << s(ii) << std::endl;
  // }
  
  // std::cout << "qall: \n"  << qall.row(256) << std::endl;
  // std::cout << "Q0all: \n" << Q0all(arma::span(0,10)) << std::endl;
  // Get quantile functions by numerical integraion, then densities by inverse of quantile density
  arma::mat Qall;
  for (uint j=0; j < r; j++) {
    if (j == 181 || j == 19) {
      arma::mat ct = cumtrapz(t.t(), qall.row(j));
      // std::cout << "cum: \n"  << j << " -- " << Q0all(j) << " -- " << ct(0, arma::span(0,5)) << std::endl;
    }
    Qall = arma::join_vert(Qall, Q0all(j) + cumtrapz(t.t(), qall.row(j)));
  }
  arma::mat fall = 1 / qall;
  // std::cout << "fall: \n" << fall(256, arma::span(0,10)) << std::endl;
  // std::cout << "Qall: \n" << Qall(256, arma::span(0,10)) << std::endl;
  
  // std::cout << "Qall rows: " << Qall.n_rows << std::endl;
  // std::cout << "ic length: " << ic.n_elem << ", k: " << k << "k+n-1: " << (k+n-1) << std::endl;
  // std::cout << "max ic value k+: " << (max(ic.subvec(k,k+n-1))) << std::endl;
  // std::cout << "max ic value k-: " << (max(ic.subvec(0,k-1))) << std::endl;
  
  // std::cout << "k: " << k << std::endl;
  // std::cout << "ic: \n" << ic.subvec(k,k+n-1) << std::endl;
  arma::mat Qfit  = Qall.rows(ic.subvec(k,k+n-1));
  // std::cout << "Qfit: \n"  << Qfit << std::endl;
  arma::mat Qpred = Qall.rows(ic.subvec(0,k-1)); 
  // std::cout << "Qpred: \n" << Qpred << std::endl;
  arma::mat qfit  = qall.rows(ic.subvec(k,k+n-1));
  // std::cout << "qfit: \n"  << qfit << std::endl;
  arma::mat qpred = qall.rows(ic.subvec(0,k-1));
  // std::cout << "qpred: \n" << qpred << std::endl;
  arma::mat ffit  = fall.rows(ic.subvec(k,k+n-1));
  // std::cout << "ffit: \n"  << ffit << std::endl;
  arma::mat fpred = fall.rows(ic.subvec(0,k-1)); 
  // std::cout << "fpred: \n" << fpred << std::endl;
  
  regression_struct result;
  result.xfit = xfit;
  result.xpred = xpred;
  result.Qfit = Qfit;
  result.Qpred = Qpred;
  result.qfit = qfit;
  result.qpred = qpred;
  result.ffit = ffit;
  result.fpred = fpred;
  result.QP_used = QP_used;
  return result;
}




}

#endif