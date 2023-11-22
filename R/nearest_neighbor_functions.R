
#' Nearest neighbors with ordering
#'
#' From a matrix of locations and number of neighbors \code{m}
#' return the matrix of the nearest \code{m} locations that
#' appear earlier in the ordering
#' @param locs A matrix with \code{n} rows and \code{d} columns.
#' Each row of locs is a point in R^d.
#' @param m the Number of neighbors
#' @param precision Choice of floating point precision
#' @return A matrix with \code{n} rows and \code{m+1} columns, with row
#' i containing the indicies of the \code{m+1} locations closest to
#' observation i that appear earlier in the ordering
#' @export
nearest_neighbors <- function(locs, m, precision = "double") {
  if (is.null(ncol(locs)) ){
        locs <- as.matrix(locs)
  }
  
  if (precision == "single") {
    return(nearest_neighbors_gpu_single(locs, m));
  } else if (precision == "double") {
    return(nearest_neighbors_gpu(locs, m));
  } else {
    stop("precision type not supported")
  }
}