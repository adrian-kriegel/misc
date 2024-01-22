/**
 * This is a benchmark on how using the Cholesky composition might sometimes be 
 * faster than explicit matrix inversion.
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>

#include <Eigen/Dense>

typedef double S;

const uint iterations = 10000;

struct Timer
{
	std::string mDescription;

	std::chrono::_V2::system_clock::time_point mStart;

	Timer(std::string description) :
	mDescription(description),
	mStart(std::chrono::high_resolution_clock::now())
	{}

	void report()
	{
		std::chrono::duration<double, std::milli> duration = 
			std::chrono::high_resolution_clock::now() - 
			mStart
		;

		std::cout << mDescription << ": " << duration.count() << " ms" << std::endl; 
	}
};

Eigen::MatrixX<S> generateRandomPositiveDefinite(uint dims)
{
	Eigen::MatrixX<S> l = Eigen::MatrixX<S>::Random(dims, dims).triangularView<Eigen::Lower>();


	Eigen::MatrixX<S> m = l * l.transpose() + Eigen::MatrixX<S>::Identity(dims, dims);

	return m;
}

void benchmarkKalmanFilter(uint stateDims, uint constraintDims)
{
	Eigen::VectorX<S> deltaMean;
	Eigen::MatrixX<S> deltaCov;

	Eigen::MatrixX<S> Cx = generateRandomPositiveDefinite(stateDims);

	Eigen::MatrixX<S> constraintCov = generateRandomPositiveDefinite(constraintDims);

	Eigen::VectorX<S> constraintMean = Eigen::VectorX<S>::Random(constraintDims);

	Eigen::MatrixX<S> J = Eigen::MatrixX<S>(constraintDims, stateDims);

	// Lz is required anyway as Mahalanobis norm is computed before every state update
	Eigen::MatrixX<S> LzFull = constraintCov.llt().matrixL();
	auto Lz = LzFull.triangularView<Eigen::Lower>();

	Timer t1("State update with Cholesky (not counting Cholesky decomposition)");

	for (uint i = 0; i < iterations; ++i)
	{
		Eigen::VectorX<S> normalizedConstraintMean = Lz.solve(constraintMean);

		deltaMean = - Cx * J.transpose() * Lz.transpose().solve(normalizedConstraintMean);

		Eigen::MatrixX<S> deltaL = Lz.solve(J * Cx);

		deltaCov = -deltaL.transpose() * deltaL;
	}

	t1.report();

	Timer t3("State update with Cholesky (counting Cholesky decomposition)");

	for (uint i = 0; i < iterations; ++i)
	{
		Eigen::MatrixX<S> LzFull = constraintCov.llt().matrixL();
		auto Lz = LzFull.triangularView<Eigen::Lower>();

		Eigen::VectorX<S> normalizedConstraintMean = Lz.solve(constraintMean);

		deltaMean = - Cx * J.transpose() * Lz.transpose().solve(normalizedConstraintMean);

		Eigen::MatrixX<S> deltaL = Lz.solve(J * Cx);

		deltaCov = -deltaL.transpose() * deltaL;
	}

	t3.report();

	Timer t2("State update classic");

	for (uint i = 0; i < iterations; ++i)
	{
		Eigen::MatrixX<S> K = Cx * J.transpose() * constraintCov.inverse();

		deltaMean = K * constraintMean;
		deltaCov = - K * J * Cx;
	}

	t2.report();
}

void benchmarkMahalanobis(int dim)
{
	Eigen::MatrixX<S> m = generateRandomPositiveDefinite(dim);

	Timer t1("using .inverse()");

	for (uint i = 0; i < iterations; ++i)
	{
		Eigen::VectorX<S> x = Eigen::VectorX<S>::Random(m.rows());
		S squaredNorm = x.dot(m.inverse() * x);
	}

	t1.report();

	Timer t2("Using forward substitution.");

	for (uint i = 0; i < iterations; ++i)
	{
		Eigen::VectorX<S> x = Eigen::VectorX<S>::Random(m.rows());

		S squaredNorm = m.llt().matrixL().solve(x).squaredNorm();
	}

	t2.report();
}

/** @returns M^{-1}B (using LLT decomposition). */
Eigen::MatrixX<S> inverseProductLLT(const Eigen::MatrixX<S> m, const Eigen::MatrixX<S> b)
{
	Eigen::LLT<Eigen::MatrixX<S>> llt = m.llt();
	return llt.matrixL().transpose().solve(llt.matrixL().solve(b).eval());
}

/** @returns M^{-1}B (using m.inverse()*b) */
Eigen::MatrixX<S> inverseProduct(const Eigen::MatrixX<S> m, const Eigen::MatrixX<S> b)
{
	return m.inverse() * b;
}

/** @returns inverse of m */
Eigen::MatrixX<S> inverseLLT(const Eigen::MatrixX<S> m)
{
	return inverseProductLLT(m, Eigen::MatrixX<S>::Identity(m.rows(), m.cols()));
}

void benchmarkInverseProduct(int dimM, int dimB)
{
	Timer t1("using .inverse()");

	Eigen::MatrixX<S> m = generateRandomPositiveDefinite(dimM);

	for (uint i = 0; i < iterations; ++i)
	{
		Eigen::MatrixX<S> b = Eigen::MatrixX<S>::Random(dimM, dimB);
		Eigen::MatrixX<S> a = inverseProduct(m, b);
	}

	t1.report();

	Timer t3("Using forward substitution.");

	for (uint i = 0; i < iterations; ++i)
	{
		Eigen::MatrixX<S> b = Eigen::MatrixX<S>::Random(dimM, dimB);
		Eigen::MatrixX<S> a = inverseProductLLT(m, b);
	}

	t3.report();

	Timer t4("Calculating LLT decomposititon.");

	for (uint i = 0; i < iterations; ++i)
	{
		Eigen::MatrixX<S> l = m.llt().matrixL();
	}

	t4.report();

	Timer t5("Performing FS.");

	Eigen::LLT<Eigen::MatrixX<S>> llt = m.llt();
	Eigen::MatrixX<S> b = Eigen::MatrixX<S>::Random(dimM, dimB);

	for (uint i = 0; i < iterations; ++i)
	{
		llt.matrixL().solve(b);
	}

	t5.report();
}

void benchmarkInverse(int dimM)
{
	Timer t1("using .inverse()");

	Eigen::MatrixX<S> m = generateRandomPositiveDefinite(dimM);

	for (uint i = 0; i < iterations; ++i)
	{
		Eigen::MatrixX<S> a = m.inverse();
	}

	t1.report();

	Timer t2("using inverseLLT(m)");

	for (uint i = 0; i < iterations; ++i)
	{
		Eigen::MatrixX<S> a = inverseLLT(m);
	}

	t2.report();
}

/** Tests that inverseProductLLT and inverseProduct produce the same result. */
void testInverseProductLLT(int dimM, int dimB)
{
	Eigen::MatrixX<S> m = generateRandomPositiveDefinite(dimM);

	Eigen::MatrixX<S> b = Eigen::MatrixX<S>(dimM, dimB);
	Eigen::MatrixX<S> a1 = inverseProduct(m, b);
	Eigen::MatrixX<S> a2 = inverseProductLLT(m, b);

	S difference = (a1 - a2).cwiseAbs().maxCoeff();

	std::cout << "Difference between inverseProductLLT inverseProduct: " << difference << std::endl;
}

int main(int argc, char* argv[])
{
	if (argc < 6)
	{
		std::cerr << "Usage: benchmark <Mahalanobis dim> <EKF state dim> <EKF constraint dim> <dim(M)> <dim(B)>" << std::endl;

		return -1;
	}


	std::cout << "# Mahalanobis norm" << std::endl;
	benchmarkMahalanobis(std::atoi(argv[1]));
	
	std::cout << "# EKF update" << std::endl;
	benchmarkKalmanFilter(std::atoi(argv[2]), std::atoi(argv[3]));

	std::cout << "# A = M^{-1}B" << std::endl;
	benchmarkInverseProduct(std::atoi(argv[4]), std::atoi(argv[5]));

	std::cout << "# A = M^{-1}" << std::endl;
	benchmarkInverse(std::atoi(argv[4]));

	testInverseProductLLT(std::atoi(argv[4]), std::atoi(argv[5]));
}

