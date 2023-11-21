#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

extern "C"
int* nearest_neighbors(float* locs, int m, int n, int dim);

extern "C"
int* nearest_neighbors_sing(float* locs, int m, int n, int dim, int nq);

// [[Rcpp::export]]
NumericMatrix nearest_neighbors_gpu(arma::mat locs, int m){
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

// [[Rcpp::export]]
NumericMatrix nearest_neighbors_sing_gpu(arma::mat locs, int m, int nq){
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

    int* NNarrayl = nearest_neighbors_sing(locsl, m, n, dim, nq);
    // arma::mat NNarray = arma::mat(&NNarrayl[0], m + 1, n, false);
    NumericMatrix NNarray(n, m + 1);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m + 1; j++){
			NNarray(i, j) = NNarrayl[i * (m + 1) + j];
		}
	}
	return NNarray;
}