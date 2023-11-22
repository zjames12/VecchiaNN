#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

extern "C"
int* nearest_neighbors(double* locs, int m, int n, int dim);

extern "C"
int* nearest_neighbors_single(float* locs, int m, int n, int dim);

//' Nearest neighbors with ordering (double precision)
//'
//' From a matrix of locations and number of neighbors \code{m}
//' return the matrix of the nearest \code{m} locations that
//' appear earlier in the ordering
//' @param locs A matrix with \code{n} rows and \code{d} columns.
//' Each row of locs is a point in R^d.
//' @param m the Number of neighbors
//' @return A matrix with \code{n} rows and \code{m+1} columns, with row
//' i containing the indicies of the \code{m+1} locations closest to
//' observation i that appear earlier in the ordering
//' @export
// [[Rcpp::export]]
NumericMatrix nearest_neighbors_gpu(arma::mat locs, int m){
    int n = locs.n_rows;
    int dim = locs.n_cols;

    // locs = locs.t();
    // double* locsl = locs.memptr();
    double* locsl = (double*) malloc(sizeof(double) * n * dim);
    for (int i = 0; i < n; i++) {
		for (int j = 0; j < dim; j++){
			locsl[i * dim + j] = locs(i, j);
		}
	}

    int* NNarrayl = nearest_neighbors(locsl, m, n, dim);
    // arma::mat NNarray = arma::mat(&NNarrayl[0], m + 1, n, false);
    NumericMatrix NNarray(n, m + 1);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m + 1; j++){
			NNarray(i, j) = NNarrayl[i * (m + 1) + j];
		}
	}
	return NNarray;
}

//' Nearest neighbors with ordering (single precision)
//'
//' From a matrix of locations and number of neighbors \code{m}
//' return the matrix of the nearest \code{m} locations that
//' appear earlier in the ordering
//' @param locs A matrix with \code{n} rows and \code{d} columns.
//' Each row of locs is a point in R^d.
//' @param m the Number of neighbors
//' @return A matrix with \code{n} rows and \code{m+1} columns, with row
//' i containing the indicies of the \code{m+1} locations closest to
//' observation i that appear earlier in the ordering
//' @export
// [[Rcpp::export]]
NumericMatrix nearest_neighbors_gpu_single(arma::mat locs, int m){
    int n = locs.n_rows;
    int dim = locs.n_cols;

    // locs = locs.t();
    // double* locsl = locs.memptr();
    float* locsl = (float*) malloc(sizeof(float) * n * dim);
    for (int i = 0; i < n; i++) {
		for (int j = 0; j < dim; j++){
			locsl[i * dim + j] = locs(i, j);
		}
	}

    int* NNarrayl = nearest_neighbors_single(locsl, m, n, dim);
    // arma::mat NNarray = arma::mat(&NNarrayl[0], m + 1, n, false);
    NumericMatrix NNarray(n, m + 1);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m + 1; j++){
			NNarray(i, j) = NNarrayl[i * (m + 1) + j];
		}
	}
	return NNarray;
}

