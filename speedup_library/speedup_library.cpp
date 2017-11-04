/*
 * speedup_library2.cpp
 *
 *  Created on: Nov 3, 2017
 *      Author: MIN
 */

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <vector>
#include <math.h>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <GenEigsSolver.h>
#include <SymEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>
#include <MatOp/DenseGenMatProd.h>
#include <MatOp/SparseSymMatProd.h>
#include <MatOp/DenseSymMatProd.h>

namespace p = boost::python;
namespace np = boost::python::numpy;
using namespace std;
using namespace Eigen;
using namespace Spectra;

//やっぱりやってたら必要になってきてるんで、実装という感じですT_T
// dist_matrix: nxn zeros
void distance_matrix_all(np::ndarray& matrix, np::ndarray& squared_dist_matrix) {
	const int n = matrix.shape(0);
	const int shape_1 = matrix.shape(1);
	auto strides = matrix.get_strides();
	auto strides_squared = squared_dist_matrix.get_strides();

	boost::multi_array<double, 2> matrix_marray(boost::extents[n][shape_1]);

	// まずmatrixをコピー(実際はコピーじゃないかもしれないけど)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < shape_1; j++) {
			matrix_marray[i][j] =
					*reinterpret_cast<double *>(matrix.get_data() + i * strides[0]
									+ j * strides[1]);
		}
	}

	boost::multi_array<double, 2> squared_dist_array(boost::extents[n][n]);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) squared_dist_array[i][j] = 0;
			else {
				double dist = 0;
				for (int k = 0; k < shape_1; k++) {
					dist += pow((matrix_marray[i][k] - matrix_marray[j][k]), 2);
				}
				squared_dist_array[i][j] = dist;
			}
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			*reinterpret_cast<double *>(squared_dist_matrix.get_data() + i * strides_squared[0]
				+ j * strides_squared[1])
				= squared_dist_array[i][j];
		}
	}
}

int ismajority(int& i, int window) {
	if (i > window) return 1;
	else return 0;
}

void raster_scan(np::ndarray& py_2d) {
	//raster scan
	const int shape_0 = py_2d.shape(0);
	const int shape_1 = py_2d.shape(1);
	auto strides_moto = py_2d.get_strides();

	for (int i = 1; i < shape_0; i++) {
		for (int j = 1; j < shape_1; j++) {
			//成分1個1個をcastする感じ
			*reinterpret_cast<int *>(py_2d.get_data() + i * strides_moto[0]
				+ j * strides_moto[1])
			+= *reinterpret_cast<int *>(py_2d.get_data() + (i-1) * strides_moto[0]
					+ (j-1) * strides_moto[1]);

		}
	}
}

void majority_vote(np::ndarray& py_2d, int window) {
	raster_scan(py_2d);

	const int shape_0 = py_2d.shape(0);
	const int shape_1 = py_2d.shape(1);
	boost::multi_array<int, 2> c_2d(boost::extents[shape_0][shape_1]);
	auto strides_moto = py_2d.get_strides();

	for (int i = 0; i < shape_0; i++) {
		for (int j = 0; j < shape_1; j++) {
			c_2d[i][j] = *reinterpret_cast<int *>(py_2d.get_data() + i * strides_moto[0]
						+ j * strides_moto[1]);
		}
	}

	const int new_shape0 = shape_0 - 2*window;
	const int new_shape1 = shape_1 - 2*window;
	const p::tuple shape = p::make_tuple(new_shape0, new_shape1);
	np::dtype dt = np::dtype::get_builtin<int>();
	np::ndarray v = np::zeros(shape, dt);
	auto strides_new = v.get_strides();

	auto majority = [](int a, int b) {
		if(a > b) return 1;
		else return 0;
	};

	for (int i = 0; i < new_shape0; i++) {
		for (int j = 0; j < new_shape1; j++ ) {
			*reinterpret_cast<int *>(v.get_data() + i * strides_new[0]
							+ j * strides_new[1])
			= majority((c_2d[i+2*window][j+2*window] - c_2d[i][j]), window);
		}
	}

	// c++ではスライスもできないし、まるごとコピーするのも無理っぽいんで、いったんきれいにしよう
	auto strides_py_2d = py_2d.get_strides();
	for (int i = 0; i < new_shape0; i++) {
		for (int j = 0; j < new_shape1; j++ ) {
			*reinterpret_cast<int *>(py_2d.get_data() + i * strides_py_2d[0]
							+ j * strides_py_2d[1])
			= *reinterpret_cast<int *>(v.get_data() + i * strides_new[0]
							+ j * strides_new[1]);
		}
	}
}

void srep(np::ndarray& dist_squared, double sigma) {
	auto shape_0 = dist_squared.shape(0);
	auto shape_1 = dist_squared.shape(1);
	auto strides = dist_squared.get_strides();

	for (int i = 0; i < shape_0; i++) {
		for (int j = 0; j < shape_1; j++ ) {
			double sq_dist_ij = *reinterpret_cast<double *>(dist_squared.get_data() + i * strides[0]
							+ j * strides[1]);
			*reinterpret_cast<double *>(dist_squared.get_data() + i * strides[0]
										+ j * strides[1])
			= exp(-(1/(2*sigma))*sq_dist_ij);
		}
	}
}

void fast_elementwise_prod(np::ndarray& a,
		np::ndarray& b, np::ndarray& result) {
	auto stride_a = a.get_strides();
	auto stride_b = b.get_strides();
	auto stride_result = result.get_strides();

	//for (int i = 0; )
}

// 正方行列xdiag(1次元行列)
void matmul_diag_a(np::ndarray& diag, np::ndarray& a, np::ndarray& result) {
	auto stride_a = a.get_strides();
	auto stride_d = diag.get_strides();
	auto stride_r = result.get_strides();
	const int n = a.shape(0);

	boost::multi_array<double, 1> diag_array(boost::extents[n]);
	for (int i = 0; i < n; i++) {
		diag_array[i] =
				*reinterpret_cast<double* >(diag.get_data() + i * stride_d[0]);
	}

	for (int i = 0 ; i < n; i++) {
		for (int j = 0; j < n; j++) {
			*reinterpret_cast<double *>(result.get_data() + i * stride_r[0]
				+ j * stride_r[1])
			= (*reinterpret_cast<double *>(a.get_data() + i * stride_a[0]
					+ j * stride_a[1]) ) // a_ik
					* diag_array[i];
		}
	}
}

void matmul_a_diag(np::ndarray& a, np::ndarray& diag, np::ndarray& result) {
	auto stride_a = a.get_strides();
	auto stride_d = diag.get_strides();
	auto stride_r = result.get_strides();
	const int n = a.shape(0);

	boost::multi_array<double, 1> diag_array(boost::extents[n]);
	for (int i = 0; i < n; i++) {
		diag_array[i] =
				*reinterpret_cast<double* >(diag.get_data() + i * stride_d[0]);
	}

	for (int i = 0 ; i < n; i++) {
		for (int j = 0; j < n; j++) {
			*reinterpret_cast<double *>(result.get_data() + i * stride_r[0]
				+ j * stride_r[1])
			= (*reinterpret_cast<double *>(a.get_data() + i * stride_a[0]
					+ j * stride_a[1]) ) // a_ik
					* diag_array[j];
		}
	}
}

np::ndarray eigenvector(np::ndarray& A) {
	// 固有ベクトルの数
	const int m = 10;

	auto shape = A.get_shape();
	const int n = shape[0];
	auto strides = A.get_strides();

	MatrixXd a(n, n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double a_ij = *reinterpret_cast<double* >(A.get_data()
					+ i * strides[0] + j * strides[1]);
			//if (a_ij == 0) continue;
			//a.insert(i, j) = a_ij;
			a(i, j) = a_ij;
			//cout << a_ij << endl;
		}
	}
	Spectra::DenseGenMatProd<double> op(a);
	//eigs(%op, 10より大きい数、（2番目の引数）*2+1より大きい数)
	Spectra::SymEigsSolver<double, SMALLEST_MAGN, Spectra::DenseGenMatProd<double> > eigs(&op, 20, 60);
	eigs.init();
	int nconv = eigs.compute();

	Eigen::MatrixXd eigvec = eigs.eigenvectors(m);

	const p::tuple shape_eig = p::make_tuple(n, m);
	np::dtype dt = np::dtype::get_builtin<double>();
	np::ndarray y = np::zeros(shape_eig, dt);
	auto y_strides = y.get_strides();
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			*reinterpret_cast<double* > (y.get_data() + i * y_strides[0]
					+ j * y_strides[1])
							= eigvec(i, j);

		}
	}
	return y;
}

BOOST_PYTHON_MODULE(speedup_library) {
	Py_Initialize();
	np::initialize();

	def("raster_scan", &raster_scan);
	def("majority_vote", &majority_vote);
	def("distance_matrix_all", &distance_matrix_all);
	def("srep", &srep);
	def("matmul_a_diag", &matmul_a_diag);
	def("matmul_diag_a", &matmul_diag_a);
	def("eigenvector", &eigenvector);
}





