// qp_example.cpp

#include <iostream>
#include <iterator>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include "AlglibSolvers.h"
#include "WassersteinRegression.h"
#include "ConfidenceBand.h"




// [[Rcpp::export]]
arma::mat linear_solver(arma::mat A, arma::mat B) {
  return wass::linear_solver(A, B);
}


// [[Rcpp::export]]
arma::vec quadprog(arma::mat A, arma::vec b, arma::mat C, arma::vec d, arma::vec lb, arma::vec ub) {
  // return wass::quadprog(wass::DENSE_AUL,A,b,C,d,lb,ub);
  return wass::quadprog(A,b,C,d,lb,ub);
}






// [[Rcpp::export]]
Rcpp::List wasser(const arma::mat xfit, const arma::mat q, const arma::mat Q0, 
                 const arma::mat xpred, const arma::vec t, const double qdmin) {
  wass::regression_struct result = wass::wasser(xfit, q, Q0, xpred, t, qdmin);
  return Rcpp::List::create(
    Rcpp::Named("q")       = q,
    Rcpp::Named("Q0")      = Q0,
    Rcpp::Named("t")       = t,
    Rcpp::Named("qdmin")   = qdmin,
    Rcpp::Named("xfit")    = result.xfit,
    Rcpp::Named("xpred")   = result.xpred,
    Rcpp::Named("Qfit")    = result.Qfit,
    Rcpp::Named("Qpred")   = result.Qpred,
    Rcpp::Named("qfit")    = result.qfit,
    Rcpp::Named("qpred")   = result.qpred,
    Rcpp::Named("ffit")    = result.ffit,
    Rcpp::Named("fpred")   = result.fpred,
    Rcpp::Named("QP_used") = result.QP_used
  );
}



// [[Rcpp::export]]
Rcpp::List confidence_band(const arma::mat xfit, const arma::mat xpred, const arma::mat Q_obs, 
                     const arma::mat q_obs, const arma::vec t_vec, const double alpha) {
  wass::confidence_struct result = wass::confidence_band(xfit, xpred, Q_obs, q_obs, t_vec, alpha);
  return Rcpp::List::create(
    Rcpp::Named("xfit")   = xfit,
    Rcpp::Named("xpred")  = xpred,
    Rcpp::Named("Q_obs")  = Q_obs,
    Rcpp::Named("t_vec")  = t_vec,
    Rcpp::Named("alpha")  = alpha,
    Rcpp::Named("Qpred")  = result.Qpred,
    Rcpp::Named("Q_lx")   = result.Q_lx,
    Rcpp::Named("Q_ux")   = result.Q_ux,
    Rcpp::Named("fpred")  = result.fpred
  );
}





