/**
 * Self-Organizing Maps on a cluster
 *  Copyright (C) 2013 Peter Wittek
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <string>
using namespace std;

#ifndef SOMOCLU_H
#define SOMOCLU_H

#if HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef _WIN32
#include<algorithm>
#endif

#define DENSE_CPU 0
#define DENSE_GPU 1
#define SPARSE_CPU 2

/// The neighbor_fuct value below which we consider
/// the impact zero for a given node in the map
#define NEIGHBOR_THRESHOLD 0.05

/// Sparse structures and routines
struct svm_node {
    int index;
    float value;
};


float euclideanDistanceOnToroidMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y, const unsigned int nSomX, const unsigned int nSomY);
float euclideanDistanceOnPlanarMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y);
float euclideanDistanceOnHexagonalToroidMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y, const unsigned int nSomX, const unsigned int nSomY);
float euclideanDistanceOnHexagonalPlanarMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y);
float getWeight(float distance, float radius, float scaling, bool compact_support, bool gaussian);
int saveCodebook(string cbFileName, float *codebook,
                 unsigned int nSomX, unsigned int nSomY, unsigned int nDimensions);
float *calculateUMatrix(float* uMatrix, float *codebook, unsigned int nSomX,
                        unsigned int nSomY, unsigned int nDimensions,
                        string mapType, string gridType);
int saveUMatrix(string fname, float *uMatrix, unsigned int nSomX,
                unsigned int nSomY);
int saveBmus(string filename, int *bmus, unsigned int nSomX,
             unsigned int nSomY, unsigned int nVectors);
float *readMatrix(const string inFilename,
                  unsigned int &nRows, unsigned int &nCols);
void readSparseMatrixDimensions(const string filename, unsigned int &nRows,
                                unsigned int &nColumns);
svm_node** readSparseMatrixChunk(const string filename, unsigned int nRows,
                                 unsigned int nRowsToRead,
                                 unsigned int rowOffset);
void trainOneEpoch(int itask, float *data, svm_node **sparseData,
                   float *codebook, int *globalBmus,
                   unsigned int nEpoch, unsigned int currentEpoch,
                   unsigned int nSomX, unsigned int nSomY,
                   unsigned int nDimensions, unsigned int nVectors,
                   unsigned int nVectorsPerRank,
                   unsigned int radius0, unsigned int radiusN,
                   string radiusCooling,
                   float scale0, float scaleN,
                   string scaleCooling,
                   unsigned int kernelType, string mapType,
                   string gridType, bool compact_support, bool gaussian);
void train(float *data, int data_length,
           unsigned int nEpoch,
           unsigned int nSomX, unsigned int nSomY,
           unsigned int nDimensions, unsigned int nVectors,
           unsigned int radius0, unsigned int radiusN,
           string radiusCooling,
           float scale0, float scaleN,
           string scaleCooling,
           unsigned int kernelType, string mapType,
           string gridType, bool compact_support, bool gaussian,
           float* codebook, int codebook_size,
           int* globalBmus, int globalBmus_size,
           float* uMatrix, int uMatrix_size);
void train(int itask, float *data, svm_node **sparseData,
           float *codebook, int *globalBmus, float *uMatrix,
           unsigned int nSomX, unsigned int nSomY,
           unsigned int nDimensions, unsigned int nVectors,
           unsigned int nVectorsPerRank, unsigned int nEpoch,
           unsigned int radius0, unsigned int radiusN,
           string radiusCooling,
           float scale0, float scaleN,
           string scaleCooling,
           unsigned int kernelType, string mapType,
           string gridType, bool compact_support, bool gaussian
#ifdef CLI
           , string outPrefix, unsigned int snapshots);
#else
          );
#endif

void trainOneEpochDenseCPU(int itask, float *data, float *numerator,
                           float *denominator, float *codebook,
                           unsigned int nSomX, unsigned int nSomY,
                           unsigned int nDimensions, unsigned int nVectors,
                           unsigned int nVectorsPerRank, float radius,
                           float scale, string mapType,
                           string gridType, bool compact_support, bool gaussian,
                           int *globalBmus);
void trainOneEpochSparseCPU(int itask, svm_node **sparseData, float *numerator,
                            float *denominator, float *codebook,
                            unsigned int nSomX, unsigned int nSomY,
                            unsigned int nDimensions, unsigned int nVectors,
                            unsigned int nVectorsPerRank, float radius,
                            float scale, string mapType,
                            string gridType, bool compact_support, bool gaussian,
                            int *globalBmus);
void initializeCodebook(unsigned int seed, float *codebook, unsigned int nSomX,
                        unsigned int nSomY, unsigned int nDimensions);


extern "C" {
#ifdef CUDA
    void setDevice(int commRank, int commSize);
    void freeGpu();
    void initializeGpu(float *hostData, int nVectorsPerRank, int nDimensions, int nSomX, int nSomY);
    void trainOneEpochDenseGPU(int itask, float *data, float *numerator,
                               float *denominator, float *codebook,
                               unsigned int nSomX, unsigned int nSomY,
                               unsigned int nDimensions, unsigned int nVectors,
                               unsigned int nVectorsPerRank, float radius,
                               float scale, string mapType,
                               string gridType, bool compact_support, bool gaussian,
                               int *globalBmus);
#endif
    void my_abort(string err);
}
#endif  // SOMOCLU_H
