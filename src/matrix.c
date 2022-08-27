#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    return (mat->data)[row * mat->cols + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    (mat->data)[row * mat->cols + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows < 1 || cols < 1) {
        return -1;
    }
    matrix *new_mat = (matrix*) malloc(sizeof(matrix));
    if (new_mat == NULL) {
        return -2;
    }

    new_mat->data = (double*) calloc(rows * cols, sizeof(double));
    if (new_mat->data == NULL) {
        return -2;
    }

    new_mat->rows = rows;
    new_mat->cols = cols;
    new_mat->parent = NULL;
    new_mat->ref_cnt = 1;

    *mat = new_mat;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    
    if (mat == NULL) {
        return;
    }
    mat->ref_cnt--;
    if (mat->parent == NULL) {
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
        }
    } else {
        deallocate_matrix(mat->parent);
        if (mat->ref_cnt == 0) {
            free(mat);
        }
    }
    
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows < 1 || cols < 1) {
        return -1;
    }
    matrix *new_mat = (matrix*) malloc(sizeof(matrix));
    if (new_mat == NULL) {
        return -2;
    }
    new_mat->data = &(from->data[offset]);
    new_mat->rows = rows;
    new_mat->cols = cols;
    new_mat->parent = from;
    new_mat->ref_cnt = 1;
    from->ref_cnt++;
    *mat = new_mat;
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO
    __m256d val_vec = _mm256_set1_pd(val);
    int end = mat->rows * mat->cols / 4 * 4;

    #pragma omp parallel for    
    for (int i = 0; i < end; i += 4) {
        _mm256_storeu_pd(mat->data + i, val_vec);
    }

    #pragma omp parallel for
    for (int i = end; i < mat->rows * mat->cols; i++) {
        mat->data[i] = val;
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO

    int end = mat->rows * mat->cols / 4 * 4;
    __m256d neg_vec = _mm256_set1_pd(-1.0);

    #pragma omp parallel for
    for (int i = 0; i < end; i += 4) {
        __m256d val_vec = _mm256_loadu_pd(mat->data + i);
        _mm256_storeu_pd(result->data + i, _mm256_max_pd(_mm256_mul_pd(val_vec, neg_vec), val_vec));
    }

    #pragma omp parallel for
    for (int i = end; i < mat->rows * mat->cols; i++) {
        result->data[i] = fabs(mat->data[i]);
    }

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int end = mat1->rows * mat1->cols / 4 * 4;

    #pragma omp parallel for
    for (int i = 0; i < end; i += 4) {
        _mm256_storeu_pd(result->data + i, _mm256_add_pd(_mm256_loadu_pd(mat1->data+i), _mm256_loadu_pd(mat2->data+i)));
    }

    #pragma omp parallel for
    for (int i = end; i < mat1->rows * mat1->cols; i++) {
        result->data[i] = mat1->data[i] + mat2->data[i];
    }

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.6 TODO
    
    
    /*
    int total;
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < mat1->rows; i++) {
        for (int k = 0; k < mat2->cols; k++) {
            total = 0;
            for (int j = 0; j < mat1->cols; j++) {
                total += get(mat1, i, j) * get(mat2, j, k);
            }
            set(result, i, k, total);
            //printf("i, k: %f ", get(result, i, k));
        }
    }
    
    */
    matrix *mat_trans = NULL;
    allocate_matrix(&mat_trans, mat2->cols, mat2->rows);
    #pragma omp parallel for collapse (2)
    for (int i = 0; i < mat2->rows; i++) {
        for (int k = 0; k < mat2->cols; k++) {
            set(mat_trans, k, i, get(mat2, i, k));
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < result->rows; i++) {
        for (int k = 0; k < result->cols; k++) {
            __m256d vec = _mm256_set1_pd(0);
            
            
            for (unsigned int j = 0; j < mat1->cols / 4 * 4; j+=4) {
                __m256d mat1_row = _mm256_loadu_pd(mat1->data + i * (mat1->cols) + j);
                __m256d mat2_row = _mm256_loadu_pd(mat_trans->data + k * (mat_trans->cols) + j);
                vec = _mm256_fmadd_pd(mat1_row, mat2_row, vec);
            }
            
            /*
            
            for (unsigned int j = 0; j < mat1->cols / 16 * 16; j += 16) {
                __m256d mat1_row = _mm256_loadu_pd(mat1->data + i * (mat1->cols) + j);
                __m256d mat2_row = _mm256_loadu_pd(mat_trans->data + k * (mat_trans->cols) + j);
                vec = _mm256_fmadd_pd(mat1_row, mat2_row, vec);

                mat1_row = _mm256_loadu_pd(mat1->data + i * (mat1->cols) + j + 4);
                mat2_row = _mm256_loadu_pd(mat_trans->data + k * (mat_trans->cols) + j + 4);
                vec = _mm256_fmadd_pd(mat1_row, mat2_row, vec);

                mat1_row = _mm256_loadu_pd(mat1->data + i * (mat1->cols) + j + 8);
                mat2_row = _mm256_loadu_pd(mat_trans->data + k * (mat_trans->cols) + j + 8);
                vec = _mm256_fmadd_pd(mat1_row, mat2_row, vec);

                mat1_row = _mm256_loadu_pd(mat1->data + i * (mat1->cols) + j + 12);
                mat2_row = _mm256_loadu_pd(mat_trans->data + k * (mat_trans->cols) + j + 12);
                vec = _mm256_fmadd_pd(mat1_row, mat2_row, vec);
            }

            for (unsigned int j = mat1->cols / 16 * 16; mat1->cols / 4 * 4; j += 4) {
                __m256d mat1_row = _mm256_loadu_pd(mat1->data + i * (mat1->cols) + j);
                __m256d mat2_row = _mm256_loadu_pd(mat_trans->data + k * (mat_trans)->cols + j);
                vec = _mm256_fmadd_pd(mat1_row, mat2_row, vec);
            }
            */
            
            double temp_arr[4];
            _mm256_storeu_pd(temp_arr, vec);
            double total = (temp_arr[0] + temp_arr[1] + temp_arr[2] + temp_arr[3]);
            
            for (int j = mat1->cols / 4 * 4; j < mat1->cols; j++) {
                total += get(mat1, i, j) * get(mat_trans, k, j);
                
            }
            //printf("%f ", total);
            set(result, i, k, total);
            
        }    
    }
    
    //printf("%d ", 212);
    deallocate_matrix(mat_trans);
    return 0;
    
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    matrix *factor1 = NULL;
    allocate_matrix(&factor1, mat->rows, mat->cols);

    matrix *factor2 = NULL;
    allocate_matrix(&factor2, mat->rows, mat->cols);
    fill_matrix(result, 0);
    #pragma omp parallel for
    for (int i = 0; i < mat->rows; i++) {
        set(result, i, i, 1);
    } 
    memcpy(factor2->data, mat->data, mat->rows * mat->cols * sizeof(double));

    while (pow > 0) {
        if (pow & 1) {
            mul_matrix(factor1, result, factor2);
            memcpy(result->data, (factor1)->data, mat->rows * mat->cols * sizeof(double));
        }
        mul_matrix(factor1, factor2, factor2);
        memcpy((factor2)->data, (factor1)->data, mat->rows * mat->cols * sizeof(double));
        pow >>= 1;
    }

    deallocate_matrix(factor1);
    deallocate_matrix(factor2);
    //printf("0, 0: %f ", get(result, 0, 0));
    
    return 0;
}
