// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// init_beta
Eigen::VectorXd init_beta(Eigen::VectorXd y, Eigen::MatrixXd X);
RcppExport SEXP _devil_init_beta(SEXP ySEXP, SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(init_beta(y, X));
    return rcpp_result_gen;
END_RCPP
}
// beta_fit
List beta_fit(Eigen::VectorXd y, Eigen::MatrixXd X, Eigen::VectorXd mu_beta, Eigen::VectorXd off, float k, int max_iter, float eps);
RcppExport SEXP _devil_beta_fit(SEXP ySEXP, SEXP XSEXP, SEXP mu_betaSEXP, SEXP offSEXP, SEXP kSEXP, SEXP max_iterSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type mu_beta(mu_betaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type off(offSEXP);
    Rcpp::traits::input_parameter< float >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< float >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(beta_fit(y, X, mu_beta, off, k, max_iter, eps));
    return rcpp_result_gen;
END_RCPP
}
// lte_n_equal_rows
bool lte_n_equal_rows(const NumericMatrix& matrix, int n, double tolerance);
RcppExport SEXP _devil_lte_n_equal_rows(SEXP matrixSEXP, SEXP nSEXP, SEXP toleranceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericMatrix& >::type matrix(matrixSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type tolerance(toleranceSEXP);
    rcpp_result_gen = Rcpp::wrap(lte_n_equal_rows(matrix, n, tolerance));
    return rcpp_result_gen;
END_RCPP
}
// get_row_groups
IntegerVector get_row_groups(const NumericMatrix& matrix, int n_groups, double tolerance);
RcppExport SEXP _devil_get_row_groups(SEXP matrixSEXP, SEXP n_groupsSEXP, SEXP toleranceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericMatrix& >::type matrix(matrixSEXP);
    Rcpp::traits::input_parameter< int >::type n_groups(n_groupsSEXP);
    Rcpp::traits::input_parameter< double >::type tolerance(toleranceSEXP);
    rcpp_result_gen = Rcpp::wrap(get_row_groups(matrix, n_groups, tolerance));
    return rcpp_result_gen;
END_RCPP
}
// make_table_if_small
List make_table_if_small(const NumericVector& x, int stop_if_larger);
RcppExport SEXP _devil_make_table_if_small(SEXP xSEXP, SEXP stop_if_largerSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type stop_if_larger(stop_if_largerSEXP);
    rcpp_result_gen = Rcpp::wrap(make_table_if_small(x, stop_if_larger));
    return rcpp_result_gen;
END_RCPP
}
// conventional_loglikelihood_fast
double conventional_loglikelihood_fast(NumericVector y, NumericVector mu, double log_theta, const arma::mat& model_matrix, bool do_cr_adj, NumericVector unique_counts, NumericVector count_frequencies);
RcppExport SEXP _devil_conventional_loglikelihood_fast(SEXP ySEXP, SEXP muSEXP, SEXP log_thetaSEXP, SEXP model_matrixSEXP, SEXP do_cr_adjSEXP, SEXP unique_countsSEXP, SEXP count_frequenciesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type log_theta(log_thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type model_matrix(model_matrixSEXP);
    Rcpp::traits::input_parameter< bool >::type do_cr_adj(do_cr_adjSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type unique_counts(unique_countsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type count_frequencies(count_frequenciesSEXP);
    rcpp_result_gen = Rcpp::wrap(conventional_loglikelihood_fast(y, mu, log_theta, model_matrix, do_cr_adj, unique_counts, count_frequencies));
    return rcpp_result_gen;
END_RCPP
}
// conventional_score_function_fast
double conventional_score_function_fast(NumericVector y, NumericVector mu, double log_theta, const arma::mat& model_matrix, bool do_cr_adj, NumericVector unique_counts, NumericVector count_frequencies);
RcppExport SEXP _devil_conventional_score_function_fast(SEXP ySEXP, SEXP muSEXP, SEXP log_thetaSEXP, SEXP model_matrixSEXP, SEXP do_cr_adjSEXP, SEXP unique_countsSEXP, SEXP count_frequenciesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type log_theta(log_thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type model_matrix(model_matrixSEXP);
    Rcpp::traits::input_parameter< bool >::type do_cr_adj(do_cr_adjSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type unique_counts(unique_countsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type count_frequencies(count_frequenciesSEXP);
    rcpp_result_gen = Rcpp::wrap(conventional_score_function_fast(y, mu, log_theta, model_matrix, do_cr_adj, unique_counts, count_frequencies));
    return rcpp_result_gen;
END_RCPP
}
// conventional_deriv_score_function_fast
double conventional_deriv_score_function_fast(NumericVector y, NumericVector mu, double log_theta, const arma::mat& model_matrix, bool do_cr_adj, NumericVector unique_counts, NumericVector count_frequencies);
RcppExport SEXP _devil_conventional_deriv_score_function_fast(SEXP ySEXP, SEXP muSEXP, SEXP log_thetaSEXP, SEXP model_matrixSEXP, SEXP do_cr_adjSEXP, SEXP unique_countsSEXP, SEXP count_frequenciesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type log_theta(log_thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type model_matrix(model_matrixSEXP);
    Rcpp::traits::input_parameter< bool >::type do_cr_adj(do_cr_adjSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type unique_counts(unique_countsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type count_frequencies(count_frequenciesSEXP);
    rcpp_result_gen = Rcpp::wrap(conventional_deriv_score_function_fast(y, mu, log_theta, model_matrix, do_cr_adj, unique_counts, count_frequencies));
    return rcpp_result_gen;
END_RCPP
}
// compute_hessian
Eigen::MatrixXd compute_hessian(Eigen::VectorXd beta, const double overdispersion, Eigen::VectorXd y, Eigen::MatrixXd design_matrix, Eigen::VectorXd size_factors);
RcppExport SEXP _devil_compute_hessian(SEXP betaSEXP, SEXP overdispersionSEXP, SEXP ySEXP, SEXP design_matrixSEXP, SEXP size_factorsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const double >::type overdispersion(overdispersionSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type design_matrix(design_matrixSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type size_factors(size_factorsSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_hessian(beta, overdispersion, y, design_matrix, size_factors));
    return rcpp_result_gen;
END_RCPP
}
// compute_scores
Eigen::MatrixXd compute_scores(Eigen::MatrixXd& design_matrix, Eigen::VectorXd& y, Eigen::VectorXd& beta, double overdispersion, Eigen::VectorXd& size_factors);
RcppExport SEXP _devil_compute_scores(SEXP design_matrixSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP overdispersionSEXP, SEXP size_factorsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type design_matrix(design_matrixSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type overdispersion(overdispersionSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type size_factors(size_factorsSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_scores(design_matrix, y, beta, overdispersion, size_factors));
    return rcpp_result_gen;
END_RCPP
}
// compute_clustered_meat
Eigen::MatrixXd compute_clustered_meat(Eigen::MatrixXd design_matrix, Eigen::VectorXd y, Eigen::VectorXd beta, double overdispersion, Eigen::VectorXd size_factors, Eigen::VectorXi clusters);
RcppExport SEXP _devil_compute_clustered_meat(SEXP design_matrixSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP overdispersionSEXP, SEXP size_factorsSEXP, SEXP clustersSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type design_matrix(design_matrixSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type overdispersion(overdispersionSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type size_factors(size_factorsSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type clusters(clustersSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_clustered_meat(design_matrix, y, beta, overdispersion, size_factors, clusters));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_devil_init_beta", (DL_FUNC) &_devil_init_beta, 2},
    {"_devil_beta_fit", (DL_FUNC) &_devil_beta_fit, 7},
    {"_devil_lte_n_equal_rows", (DL_FUNC) &_devil_lte_n_equal_rows, 3},
    {"_devil_get_row_groups", (DL_FUNC) &_devil_get_row_groups, 3},
    {"_devil_make_table_if_small", (DL_FUNC) &_devil_make_table_if_small, 2},
    {"_devil_conventional_loglikelihood_fast", (DL_FUNC) &_devil_conventional_loglikelihood_fast, 7},
    {"_devil_conventional_score_function_fast", (DL_FUNC) &_devil_conventional_score_function_fast, 7},
    {"_devil_conventional_deriv_score_function_fast", (DL_FUNC) &_devil_conventional_deriv_score_function_fast, 7},
    {"_devil_compute_hessian", (DL_FUNC) &_devil_compute_hessian, 5},
    {"_devil_compute_scores", (DL_FUNC) &_devil_compute_scores, 5},
    {"_devil_compute_clustered_meat", (DL_FUNC) &_devil_compute_clustered_meat, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_devil(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
