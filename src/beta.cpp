#include <RcppEigen.h>
#include <iostream>
#include "batch.hpp"
// [[Rcpp::depends(RcppEigen)]]
#include <Eigen/Dense>
/*
void beta_fit_gpu_external(Eigen::MatrixXf Y_host, Eigen::MatrixXf X_host,
                           Eigen::MatrixXf mu_beta_host,
                           Eigen::MatrixXf offset_host, Eigen::VectorXf k_host,
                           int max_iter, float eps);

void hello();
*/


using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::export]]
Eigen::VectorXd init_beta(Eigen::VectorXd y, Eigen::MatrixXd X) {
  VectorXd norm_log_count_mat = y.array().log1p();
  return X.colPivHouseholderQr().solve(norm_log_count_mat);
}

// [[Rcpp::export]]
List beta_fit(Eigen::VectorXd y, Eigen::MatrixXd X, Eigen::VectorXd mu_beta, Eigen::VectorXd off, float k, int max_iter, float eps) {
  int cols = X.cols();
  int rows = X.rows();
  k = 1.0 / k;
  Eigen::VectorXd delta = Eigen::VectorXd::Zero(cols);
  Eigen::MatrixXd inv_sigma_beta_const = 0.01 * Eigen::MatrixXd::Identity(cols, cols);
  Eigen::MatrixXd Zigma = Eigen::MatrixXd::Identity(cols, cols);
  VectorXd mu_g = VectorXd::Zero(rows);
  VectorXd w_q = VectorXd::Zero(rows);

  bool converged = 0;
  int iter = 0;
  while (!converged && iter < max_iter) {
    w_q = (-X * mu_beta - off).array().exp();
    mu_g = (k + y.array()) / (1 + k * w_q.array());
    Zigma = (k * X.transpose() * (mu_g.array() * w_q.array()).matrix().asDiagonal() * X).inverse();
    delta = Zigma * (k * X.transpose() * (mu_g.array() * w_q.array() - 1).matrix());
    mu_beta += delta;
    converged = delta.cwiseAbs().maxCoeff() < eps;
    iter++;
  }
  // hello();
  // Return both mu_beta and Zigma as a List
  return List::create(Named("mu_beta") = mu_beta, Named("iter") = iter);
}

// [[Rcpp::export]]
Eigen::MatrixXf  beta_fit_gpu(Eigen::MatrixXd y, Eigen::MatrixXd X, Eigen::MatrixXd mu_beta, Eigen::MatrixXd off, Eigen::VectorXd k, int max_iter, float eps,int batch_size) {
  
Eigen::MatrixXf y_float = y.cast<float>();
Eigen::MatrixXf X_float = X.cast<float>();
Eigen::MatrixXf mu_beta_float = mu_beta.cast<float>();
Eigen::MatrixXf off_float = off.cast<float>();
Eigen::VectorXf k_float = k.cast<float>();
 std::cout<<"Start GPU" << std::endl;
auto result= beta_fit_gpu_external(y_float, X_float, mu_beta_float, off_float, k_float, batch_size,
                       eps);
  std::cout<<"END GPU" << std::endl;
//  Return both mu_beta and Zigma as a List

  return  result;
}

// [[Rcpp::export]]
List beta_fit_group(Eigen::VectorXd y, float mu_beta, Eigen::VectorXd off, float k, int max_iter, float eps) {
  int rows = y.size();

  k = 1.0 / k;
  float delta = 0.0;
  float Zigma = 0.0;
  VectorXd mu_g = VectorXd::Zero(rows);
  VectorXd w_q = VectorXd::Zero(rows);
  VectorXd ones = VectorXd::Ones(rows);

  bool converged = 0;
  int iter = 0;
  while (!converged && iter < max_iter) {
    w_q = (-mu_beta * ones - off).array().exp();
    mu_g = (k + y.array()) / (1 + k * w_q.array());
    Zigma = 1.0 / (k * (mu_g.array() * w_q.array()).sum());

    delta = Zigma * (k * (mu_g.array() * w_q.array() - 1).sum());
    mu_beta += delta;
    converged = delta < eps;
    iter++;
  }

  // Return both mu_beta and Zigma as a List
  return List::create(Named("mu_beta") = mu_beta, Named("iter") = iter);
}

// Check how many unique rows are in a matrix and if this number is less than or equal to n
// This is important to determine if the model can be solved by group averages
// (ie. the numer of unique rows == number of columns)
// [[Rcpp::export]]
bool lte_n_equal_rows(const NumericMatrix& matrix, int n, double tolerance = 1e-10) {
  NumericMatrix reference_matrix(n, matrix.ncol());
  size_t n_matches = 0;
  for(size_t row_idx = 0; row_idx < matrix.nrow(); row_idx++){
    bool matched = false;
    NumericMatrix::ConstRow vec = matrix(row_idx, _);
    for(size_t ref_idx = 0; ref_idx < n_matches; ref_idx++){
      NumericMatrix::Row ref_vec  = reference_matrix(ref_idx, _);
      if(sum(abs(vec - ref_vec)) < tolerance){
        matched = true;
        break;
      }
    }
    if(! matched){
      ++n_matches;
      if(n_matches > n){
        return false;
      }
      reference_matrix(n_matches - 1, _) = vec;
    }
  }
  return true;
}

// [[Rcpp::export]]
IntegerVector get_row_groups(const NumericMatrix& matrix, int n_groups, double tolerance = 1e-10) {
  NumericMatrix reference_matrix(n_groups, matrix.ncol());
  IntegerVector groups(matrix.nrow());
  size_t n_matches = 0;
  for(size_t row_idx = 0; row_idx < matrix.nrow(); row_idx++){
    bool matched = false;
    NumericMatrix::ConstRow vec = matrix(row_idx, _);
    for(size_t ref_idx = 0; ref_idx < n_matches; ref_idx++){
      NumericMatrix::Row ref_vec  = reference_matrix(ref_idx, _);
      if(sum(abs(vec - ref_vec)) < tolerance){
        groups(row_idx) = ref_idx;
        matched = true;
        break;
      }
    }
    if(! matched){
      groups(row_idx) = n_matches;
      reference_matrix(n_matches, _) = vec;
      ++n_matches;
    }
  }
  return groups + 1;
}
// // [[Rcpp::export]]
// List beta_fit_global(Eigen::MatrixXd Y, Eigen::MatrixXd X, Eigen::VectorXd MU_BETA, int max_iter, float eps) {
//   int nfeatures = X.cols();
//   int ncells = X.rows();
//   int ngenes = Y.rows();
//
//   Eigen::MatrixXd BETA(ngenes, nfeatures);
//   std::vector<MatrixXd> SIGMA(ngenes, MatrixXd(nfeatures, nfeatures));
//
//   for (int i=0; i<ngenes; i++) {
//     List fit_result = beta_fit(Y.row(i), X, MU_BETA.row(i), max_iter, eps);
//
//     BETA.row(i) = as<VectorXd>(fit_result["mu_beta"]);
//     SIGMA[i] = as<MatrixXd>(fit_result["Zigma"]);
//
//   }
//
//   return List::create(Named("Beta") = BETA, Named("Sigma") = SIGMA);
// }
