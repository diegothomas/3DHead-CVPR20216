#ifndef __BINARY_FILE_IO_H_
#define __BINARY_FILE_IO_H_

#include <assert.h>
#include <stdio.h>
#include <stdexcept>
#include <memory>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <intraface/Macros.h>

using namespace std;

namespace facio {

	struct FileDeleter {  
		void operator()(FILE* p) {
			fclose(p);
		}
	};
	typedef unique_ptr<FILE,FileDeleter> FilePtr;

	// Check whether we have a big endian architecture
	inline bool is_bigendian()
	{
		const int i = 1;
		return (*(char*)&i) == 0;
	}

	template <typename T>
	T swap_endian(T t)
	{
		T u;
		unsigned char* t_ptr = (unsigned char *)&t;
		unsigned char* u_ptr = (unsigned char *)&u;

		for (size_t k = 0; k < sizeof(T); k++)
			u_ptr[k] = t_ptr[sizeof(T) - k - 1];

		return u;
	}


	// Write array of n elements to file
	template <typename T>
	bool write_n(FILE * file, const T * data, size_t n)
	{
		if (is_bigendian()) {
			// This is not optimized for speed, however it's only big endian writing....
			bool okay = true;
			for (size_t i = 0; i < n; ++i) {
				T swapped = swap_endian(data[i]);
				okay &= fwrite(&swapped, sizeof(swapped), 1, file) == 1;
			}
			return okay;
		} else {
			return fwrite(data, sizeof(*data), n, file) == n;
		}
	}

	// Read array of n elements from file
	template <typename T>
	bool read_n(FILE * file, T * data, size_t n)
	{
		if (fread(data, sizeof(*data), n, file) != n)
			return false;
		if (is_bigendian()) {
			for (size_t i = 0; i < n; ++i)
				data[i] = swap_endian(data[i]);
		}
		return true;
	}

	template <typename T>
	bool read_one(FILE * file, T & data)
	{
		return read_n(file, &data, 1);
	}

	// Write one element to file
	template <typename T>
	bool write_one(FILE * file, T data)
	{
		return write_n(file, &data, 1);
	}

	// Read one Eigen::Vector from file
	template<class T>
	bool read_one(FILE * file, Eigen::Matrix<T,Eigen::Dynamic,1> & data)
	{
		bool okay = true;
		int32_t rows;
		okay &= read_one(file, rows);
		if (rows <= 0)
			return false;
		data.resize(rows);
		okay &= read_n<T>(file, static_cast<T *>(data.data()), rows);
		return okay;
	}

	// Read one Eigen::Matrix from file
	template<class T>
	bool read_one(FILE * file, Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> & data)
	{
		bool okay = true;
		int32_t rows, cols;
		okay &= read_one(file, rows);
		okay &= read_one(file, cols);     
		if (rows <= 0 || cols <= 0)
			return false;

		data.resize(rows, cols);
		okay &= read_n<T>(file, static_cast<T *>(data.data()), rows*cols);
		return okay;
	}

	// Write one Eigen::Matrix to file
	template<class T>
	bool write_one(FILE * file, const Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic> & data)
	{
		bool okay = true;
		int rows = data.rows();
		int cols = data.cols();
		int total = rows * cols;
		okay &= write_one(file, int32_t(rows));
		okay &= write_one(file, int32_t(cols));

		const T *dataPtr = static_cast<const T *>(data.data());
		okay &= write_n<T>(file, dataPtr, total);
		return okay;
	}

	// Write one Eigen::Vector to file
	template<class T>
	bool write_one(FILE * file, const Eigen::Matrix<T,Eigen::Dynamic,1> & data)
	{
		bool okay = true;
		int rows = data.rows();
		okay &= write_one(file, int32_t(rows));

		const T *dataPtr = static_cast<const T *>(data.data());
		okay &= write_n<T>(file, dataPtr, rows);
		return okay;
	}

	/** \addtogroup core
	*  @{
	*/

	/*! 
	\fn bool read_one(FILE * file, cv::Mat & data)
	\brief Reads a OpenCV matrix from file.
	\param file FILE pointer.
	\param data reference to a OpenCV Mat.
	\return bool flag indicating whether operation is successful.
	*/
	DLLDIR bool read_one(FILE * file, cv::Mat & data);

	/*! 
	\fn bool read_one(FILE * file, cv::Mat & data)
	\brief Reads an Eigen matrix from file.
	\param file FILE pointer.
	\param data reference to an Eigen matrix.
	\return bool flag indicating whether operation is successful.
	*/
	//DLLDIR bool read_one(FILE * file, Eigen::MatrixXf & data);

	/*! 
	\fn bool read_one(FILE * file, Eigen::VectorXf & data)
	\brief Reads an Eigen vector from file.
	\param file FILE pointer.
	\param data reference to an Eigen vector.
	\return bool flag indicating whether operation is successful.
	*/
	//DLLDIR bool read_one(FILE * file, Eigen::VectorXf & data);

	/*! 
	\fn bool write_one(FILE * file, const cv::Mat & data)
	\brief Writes a OpenCV matrix to file.
	\param file FILE pointer.
	\param data reference to a OpenCV Mat.
	\return bool flag indicating whether operation is successful.
	*/
	DLLDIR bool write_one(FILE * file, const cv::Mat& data);

	/*! 
	\fn bool write_one(FILE * file, const Eigen::MatrixXf & data)
	\brief Writes a Eigen matrix to file.
	\param file FILE pointer.
	\param data reference to a Eigen matrix.
	\return bool flag indicating whether operation is successful.
	*/
	//DLLDIR bool write_one(FILE * file, const Eigen::MatrixXf & data);

	/*! 
	\fn FILE* open_and_check(const char *model);
	\brief open and check if it is a valid mode file.
	\param model model file name.
	\return FILE* a file pointer.
	*/
	DLLDIR FilePtr open_and_check(const char *model);

	// Write one std::string to file
	DLLDIR bool write_one(FILE * file, const string& data);

	// Read one std::string from file
	DLLDIR bool read_one(FILE * file, string& data);


	/** @}*/
}

#endif
