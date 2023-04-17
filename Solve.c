/*Matrix A - define in CSR, convert to BSR*/
/*Vector b, vector r, Vector r-o*/
/*rho, alphs, pmega, itr - Scalars*/
//rho, i-1 and rho_i, omega_i-1 and omega_i
//Steps 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include "helper_functions.h"  // helper for shared functions common to CUDA Samples
#include "helper_cuda.h"       // helper function CUDA error checking and initialization
/*Initialize CSR*/
#include "cusparse_bcgstab_4.h"



/*Converts a column-based ordering scheme to row-based
 Output : RHS_y - RHS in a row-major format*/
__global__
void XtoY(double*RHS, int Ny, int Nx, double*RHS_y){

    int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 

    if (thread_idx < Nx*Ny){
        int row = thread_idx % Ny;
        int col =  thread_idx / Ny; 

        RHS_y[thread_idx] = RHS[row*Nx+col];
    }
    
}

/*Converts a row-major ordering scheme to column-major
 Output : RHS_y - RHS in a column-major format*/
__global__
void YtoX(double*RHS, int Ny, int Nx, double*RHS_y){

    int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 

    if (thread_idx < Nx*Ny){
        int row = thread_idx / Nx;
        int col =  thread_idx % Nx; 

        RHS_y[thread_idx] = RHS[col*Ny+row];
    }
    
}



/* Solves a matrix system Ax=b
* Uses cusparse library
* A is represented in a CSR format
* I represent the row indices and J represents col indices
* Performs an in-place solve. Thus, RHS is rewritten with the solution
*/

void TriDiagSolve(double* low , double* diag , double* high, double* RHS, int rows){


    int dimBlock = 1;   //CSR = BSR wth block dimension 1


    /*Initialize the variables to be used in the linear solve*/
    cusparseHandle_t handle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&handle);
    size_t pBufferSize; 
    void* pBuffer = 0;
    int structural_zero, numerical_zero;    //To check for singularities in the coefficient matrix



    /* Allocate buffer space for linear solve*/
    checkCudaErrors(cusparseDgtsv2_bufferSizeExt(handle, rows, 1, 
            low,diag, high, RHS ,rows , &pBufferSize));  

    checkCudaErrors(cudaMalloc((void**)&pBuffer, sizeof(int)*pBufferSize));

    /*5 - Analyze coefficient matrix and report any singularities */
    checkCudaErrors(cusparseDgtsv2(handle, rows, 1, 
            low,diag, high, RHS ,rows , pBuffer)); 

}




/* Generates the coeffieicents for solving  traidiagonal equation along X
* Stores the lower, upper diagonal and main diag in three vectors
* Input:low, high and diag are the three vectors for the diagonals
* input: rows is the total number of vertices - Ny*Nx
*/
__global__
void genXCoeffs(double *low, double *high, double*diag, const vertex* Domain, 
     int rows, double mu, double rho, double delT){

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    vertex thisV = Domain[thread_idx];      //This vertex
    double dely = thisV.dely; 
    double delx = thisV.delx; 

    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
            /*Interior vertices*/
                low [thread_idx] =  - (mu/rho * (dely/delx)/(dely*delx))*(delT/2);
                high [thread_idx] =  - mu/rho * (dely/delx)/(dely*delx) * (delT/2);
                diag [thread_idx] =  1 + 2 * mu/rho * (dely/delx)/(dely*delx) * (delT/2);
                break;

            case '1':
                switch (thisV.BSide){
                    case 'L':
                    /*Left-Side Boundary*/
                        if ((thisV.BType == '0') || (thisV.BType == '2')) {
                            /*Inlet or Wall*/
                            /*The value of velocity is fixed. So the TDM coeffs are sinple [0,1,0]*/
                            low [thread_idx] =  0;
                            high [thread_idx] =  0;
                            diag [thread_idx] =  1 ;

                        } else if (thisV.BType == '1') {
                            /*Symmetry BC */
                            /*Left  = Right So, left coeff is zero and right coeff is doubled*/
                            low [thread_idx] =  0;
                            high [thread_idx] = - 1*mu/rho * (dely/delx)/(dely*delx)*(delT/2);
                            diag [thread_idx] =  1 + 1 * mu/rho * (dely/delx)/(dely*delx)* (delT/2);

                        }
                    break;

                    case 'R':
                        /*Right-Side Boundary*/
                        if ((thisV.BType == '0')|| (thisV.BType == '2')){
                            /*Inlet or Wall*/
                            low [thread_idx] =  0;
                            high [thread_idx] =  0;
                            diag [thread_idx] =  1 ;

                        } else if (thisV.BType == '1') {
                            /*Symmetry BC */
                            high [thread_idx] =  0;
                            low [thread_idx] =  - 1*mu/rho * (dely/delx)/(dely*delx) * (delT/2);
                            diag [thread_idx] =  1 + 1 * mu/rho * (dely/delx)/(dely*delx) * (delT/2);

                        }

                    break; 
                    default:
                    /*Top and bottom boundaries treated as interior zones as discretization is only along x*/
                        low [thread_idx] =  - mu/rho * (dely/delx)/(dely*delx) * (delT/2);
                        high [thread_idx] =  - mu/rho * (dely/delx)/(dely*delx) * (delT/2);
                        diag [thread_idx] =  1 + 2 * mu/rho * (dely/delx)/(dely*delx) * (delT/2); 

                }

            break;
            case '2':

                if (thisV.BType == '0'){

                    /*Inlet condition*/
                     low [thread_idx] =  0;
                     high [thread_idx] =  0;
                     diag [thread_idx] =  1;

                } else{
                    if (thisV.BSide == 'W' || thisV.BSide == 'Z' ){

                    /*Left Side points*/
                        if (thisV.BType == '1') {

                                /*Symmetry BC*/
                                low [thread_idx] =  0;
                                high [thread_idx] =  - 2*mu/rho * (dely/delx)/(dely*delx) * (delT/2);
                                diag [thread_idx] =  1 + 2 * mu/rho * (dely/delx)/(dely*delx) * (delT/2);

                        }
                    } else if ( thisV.BSide == 'X' || thisV.BSide == 'Y') {
                        if (thisV.BType == '1') {

                                /*Symmetry BC*/
                                high [thread_idx] =  0;
                                low [thread_idx] =  - 2*mu/rho * (dely/delx)/(dely*delx) *(delT/2);
                                diag [thread_idx] =  1 + 2 * mu/rho * (dely/delx)/(dely*delx) * (delT/2);

                        }
                    }  
                }

            break;

        }

    }
}

/* Generates the coeffieicents for solving  traidiagonal equation along Y
* Please note that this assumes row-based ordering. 
* However, domain is defined with a column-based order
* Postrequisite : low, high and diag changed to row-based ordering
* input: rows is the total number of vertices - Ny*Nx
*/
__global__
void genYCoeffs(double *low, double *high, double*diag, const vertex* Domain, 
     int rows, double mu, double rho, double delT){

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    vertex thisV = Domain[thread_idx]; 
    double dely = thisV.dely; 
    double delx = thisV.delx; 

    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
            /*Interior vertices*/
                low [thread_idx] =  - mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                high [thread_idx] =  - mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                diag [thread_idx] =  1 + 2 * mu/rho * (delx/dely)/(dely*delx)  * (delT/2);

                break;

            case '1':
                switch (thisV.BSide){
                    case 'B':
                    /*Bottom-Side Boundary*/
                        if ((thisV.BType == '0') || (thisV.BType == '2')) {
                            /*Inlet or Wall*/
                            /*The value of velocity is fixed. So the TDM coeffs are sinple [0,1,0]*/
                            low [thread_idx] =  0;
                            high [thread_idx] =  0;
                            diag [thread_idx] =  1 ;

                        } else if (thisV.BType == '1') {
                            /*Symmetry BC */
                            low [thread_idx] =  0;
                            high [thread_idx] =  - 1*mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                            diag [thread_idx] =  1 + 1 * mu/rho * (delx/dely)/(dely*delx) * (delT/2);

                        }
                    break;

                    case 'T':
                        /*Top-Side Boundary*/
                        if ((thisV.BType == '0')|| (thisV.BType == '2')){
                            /*Dirichlet BC*/
                            low [thread_idx] =  0;
                            high [thread_idx] =  0;
                            diag [thread_idx] =  1 ;

                        } else if (thisV.BType == '1') {
                            /*Symmetry BC */
                            high [thread_idx] =  0;
                            low [thread_idx] =  - 1*mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                            diag [thread_idx] =  1 + 1 * mu/rho * (delx/dely)/(dely*delx) * (delT/2);

                        }

                    break; 
                    default:
                    /*Left and right boundaries treated as interior zones*/
                        low [thread_idx] =  - mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                        high [thread_idx] =  - mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                        diag [thread_idx] =  1 + 2 * mu/rho * (delx/dely)/(dely*delx) * (delT/2); 

                }

            break;
            case '2':

                if (thisV.BType == '0'){

                    /*Specified value condition*/
                     low [thread_idx] =  0;
                     high [thread_idx] =  0;
                     diag [thread_idx] =  1;

                } else{
                    if (thisV.BSide == 'X' || thisV.BSide == 'Z' ){

                    /*Bottom Side points*/
                        if (thisV.BType == '1') {

                                /*Symmetry BC*/
                                low [thread_idx] =  0;
                                high [thread_idx] =  - 2*mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                                diag [thread_idx] =  1 + 2 * mu/rho * (delx/dely)/(dely*delx) * (delT/2);

                        }
                    } else if ( thisV.BSide == 'D' || thisV.BSide == 'W') {

                        /*Top Side points*/

                        if (thisV.BType == '1') {

                                /*Symmetry BC*/
                                high [thread_idx] =  0;
                                low [thread_idx] =  - 2*mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                                diag [thread_idx] =  1 + 2 * mu/rho * (delx/dely)/(dely*delx) * (delT/2);

                        }
                    }  
                }

            break;

        }

    }
}


/* Generates the convective terms to be plugged into the RHS for velocities
* Input : vec is the variable for which the  terms are needed e.g for H_u, vec is U
* Input : rows is the total no of data points (NY*Nx)
*/
__global__
void calcH(const double *U, const double *V, const double*vec, double* H, 
    const vertex* Domain, int rows, int Nx){

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    vertex thisV = Domain[thread_idx];   //This vertex
    double dely = thisV.dely; 
    double delx = thisV.delx; 

    double right;
    double left;
    double top;
    double btm;

    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
            /*Interior vertices*/
                right = ((U[thread_idx]+ U[thread_idx+1])/2)*
                        ((vec[thread_idx]+ vec[thread_idx+1])/2);
                left = ((U[thread_idx]+ U[thread_idx-1])/2)*
                        ((vec[thread_idx]+ vec[thread_idx-1])/2);  
                top = ((V[thread_idx]+ V[thread_idx-Nx])/2)*
                        ((vec[thread_idx]+ vec[thread_idx-Nx])/2);
                btm = ((V[thread_idx]+ V[thread_idx+Nx])/2)*
                        ((vec[thread_idx]+ vec[thread_idx+Nx])/2);  

                H[thread_idx] = ((right - left)*dely +
                         (top - btm )*delx)/(dely*delx);
                break;

            case '1':

                if (thisV.BType == '0' || thisV.BType == '2'){
                    /*Specified inlet or Wall*/
                    /*As value is fixed, no equation needs to be solved. So H is zero*/
                    H[thread_idx] = 0;

                } else {
                    switch (thisV.BSide){
                        case 'B':
                            /*Bottom-Side Boundary*/
                            /*Top = bottom and so they cancel out*/
                            right = ((U[thread_idx]+ U[thread_idx+1])/2)*
                                    ((vec[thread_idx]+ vec[thread_idx+1])/2);
                            left = ((U[thread_idx]+ U[thread_idx-1])/2)*
                                    ((vec[thread_idx]+ vec[thread_idx-1])/2);  

                            H[thread_idx] = ((right - left)*dely)/(dely*delx);
                        break;

                        case 'T':
                            /*Top-Side Boundary*/
                            /*Top = btm and so they cancel out*/
                            right = ((U[thread_idx]+ U[thread_idx+1])/2)*
                                    ((vec[thread_idx]+ vec[thread_idx+1])/2);
                            left = ((U[thread_idx]+ U[thread_idx-1])/2)*
                                    ((vec[thread_idx]+ vec[thread_idx-1])/2);  
                            
                            H[thread_idx] = ((right - left)*dely)/(dely*delx);
                            break;

                        case 'R':
                        /*Right-Side Boundary*/
                            top = ((V[thread_idx]+ V[thread_idx-Nx])/2)*
                                    ((vec[thread_idx]+ vec[thread_idx-Nx])/2);
                            btm = ((V[thread_idx]+ V[thread_idx+Nx])/2)*
                                    ((vec[thread_idx]+ vec[thread_idx+Nx])/2); 

                            H[thread_idx] = ( (top - btm )*delx)/(dely*delx);
                        break;

                        case 'L':
                        /*Left-Side Boundary*/
                            top = ((V[thread_idx]+ V[thread_idx-Nx])/2)*
                                  ((vec[thread_idx]+ vec[thread_idx-Nx])/2);
                            btm = ((V[thread_idx]+ V[thread_idx+Nx])/2)*
                                  ((vec[thread_idx]+ vec[thread_idx+Nx])/2);  
                            
                            H[thread_idx] = ( (top - btm )*delx)/(dely*delx);
                        break;

                    }

                break;
            }
            case '2':

                if (thisV.BType == '0' || thisV.BType == '2'){

                    /*Dirichlet condition*/
                    H[thread_idx] = 0;

                } else{
                     /*Symmetry condition*/
                    H[thread_idx] = 0;
                   
                }
            break; 
        }
    }
}


/* Generates the diffusive terms to be plugged into the RHS for velocities
* Input :carId is the variable for which diff is needed 'U' for U and 'V' for V
* Input : rows is the total number of vertexes [Ny*Nx]
*/
__global__
void calcD(const double *vec, double*D,  const vertex* Domain, int rows, int Nx, 
    char varId, double delT, double mu, double rho){

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    vertex thisV = Domain[thread_idx]; 
    double dely = thisV.dely; 
    double delx = thisV.delx; 
    double thisVar; 


    double right;
    double left;
    double top;
    double btm;

    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
            /*Interior vertices*/

                right = (vec[thread_idx+1]- vec[thread_idx])/delx;
                left = (vec[thread_idx]- vec[thread_idx-1])/delx;  
                top = (vec[thread_idx-Nx]- vec[thread_idx])/dely;
                btm = (vec[thread_idx]- vec[thread_idx + Nx])/dely; 

                D[thread_idx] = (((right - left)*dely + (top - btm )*delx)/
                                (dely*delx))*delT/2 * mu/rho + 
                                vec[thread_idx];

                break;

            case '1':

                if (thisV.BType == '0' || thisV.BType == '2'){
                    /*Inlet OR Wall*/
                    /*RHS is set to the input boundary value*/

                    switch (varId){
                        case 'U':
                        D[thread_idx] = thisV.UValue;
                        break; 
                        case 'V':
                        D[thread_idx] = thisV.VValue;
                        break; 
                    }

                } else {
                    /*Symmetry BCs*/
                    /*Left= = right or Btm  =Top depending on the boundary*/

                    switch (thisV.BSide){
                        case 'B':
                            /*Bottom-Side Boundary*/
                            /*Btm = Top*/
                            right = (vec[thread_idx+1]- vec[thread_idx])/delx;
                            left = (vec[thread_idx]- vec[thread_idx-1])/delx;  
                            top = (vec[thread_idx-Nx]- vec[thread_idx])/dely;
                            btm = (vec[thread_idx-Nx]- vec[thread_idx])/dely;

                            H[thread_idx] = ((right - left)*dely + (top - btm )*delx)/
                                            (dely*delx) * delT/2 * mu/rho + 
                                            vec[thread_idx];
                        break;

                        case 'T':
                            /*Top-Side Boundary*/
                            /*Btm = Top*/
                            right = (vec[thread_idx+1]- vec[thread_idx])/delx;
                            left = (vec[thread_idx]- vec[thread_idx-1])/delx;  
                            top = (vec[thread_idx]- vec[thread_idx + Nx])/dely;
                            btm = (vec[thread_idx]- vec[thread_idx + Nx])/dely;
                            H[thread_idx] = ((right - left)*dely + (top - btm )*delx)/
                                            (dely*delx) * delT/2 *mu/rho +
                                            vec[thread_idx];

                            break;

                        case 'R':
                        /*Right-Side Boundary*/
                        /*Left = Right*/
                            right = (vec[thread_idx]- vec[thread_idx-1])/delx;
                            left = (vec[thread_idx]- vec[thread_idx-1])/delx;  
                            top = (vec[thread_idx-Nx]- vec[thread_idx])/dely;
                            btm = (vec[thread_idx]- vec[thread_idx + Nx])/dely; 

                            H[thread_idx] = ((right - left)*dely + (top - btm )*delx)/
                                            (dely*delx) *delT/2 *mu/rho+
                                            vec[thread_idx];
                        break;

                        case 'L':
                        /*Left-Side Boundary*/
                            right = (vec[thread_idx+1]- vec[thread_idx])/delx;
                            left = (vec[thread_idx+1]- vec[thread_idx])/delx;  
                            top = (vec[thread_idx-Nx]- vec[thread_idx])/dely;
                            btm = (vec[thread_idx]- vec[thread_idx + Nx])/dely; 
                            H[thread_idx] = ((right - left)*dely + (top - btm )*delx)/
                                            (dely*delx) * delT/2 *mu/rho +
                                            vec[thread_idx] ;

                        break;

                    }

                break;
            }
            case '2':

                if (thisV.BType == '0' || thisV.BType == '2'){

                    /*Dirichlet condition*/
                    switch (varId){
                        case 'U':
                        H[thread_idx] = thisV.UValue;
                        break; 
                        case 'V':
                        H[thread_idx] = thisV.VValue;
                        break; 
                    }

                } else{
                     /*Symmetry condition*/
                    H[thread_idx] = 0;
                   
                }
            break; 
        }
    }
}

/*Adds up the convective and diffusive terms to update the RHS for TDM equations
RHS = 1/2* delT* (3*H_u_n - H_u_n_1) + D_u_n*/

void updateRHS(cublasHandle_t handleBlas, double* H_u_n_1, double* H_v_n_1,
     double* H_u_n, double* H_v_n, double* D_u_n, double* D_v_n, 
     double* RHS_u_n, double* RHS_v_n, double delT, double mu, double rho, 
     int rows ){

    double alpha = -1.5 *delT; 

    /*Add vectors - RHS_n = RHS_n - 3/2*H_n_1 */
    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                           &alpha,H_u_n_1, 1,RHS_u_n, 1));

    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                        &alpha,H_v_n_1, 1,RHS_v_n, 1));

    alpha = 0.5 *delT; 

    /*Add vectors - RHS_n = RHS_n + 1/2*H_n*/
    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                           &alpha,H_u_n, 1,RHS_u_n, 1));

    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                        &alpha,H_v_n, 1,RHS_v_n, 1));


    alpha = 1; 

    /*Add vectors - RHS_n = RHS_n + D_n*/
    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                           &alpha,D_u_n, 1,RHS_u_n, 1));

    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                        &alpha,D_v_n, 1,RHS_v_n, 1));  


    
}

/*Generates the coefficients of the pressure Poisson equation in CSR format*/
/*The basic pressure Poisson equation follows the pattern - 
*   [1 0 0 0.... 0 . . . .. . .]
     .
     .
     .1 .. 1  -4 1 ..1 0..0
     .
     .
     .0 ....0................1]


 *Stencil is [1,1,-4,1,1] and the variables are [Ptop, Pleft, P, Pright and Pbtm]
* Coeffieicnts are to be multiplied by the appropriate constsnat (Not done yet)
*/

__global__
void getPCoeffs(int* P_rowPtr, int* P_colPtr, double* P_val,
                 const vertex* Domain, int rows, int Nx, int Ny,
                 double rho, double delT){


    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int start = thread_idx*5; 
    vertex thisV = Domain[thread_idx]; 
    double delx = thisV.delx;
    double dely = thisV.dely; 
    double constTerm  = 1/(rho*(delx*dely))*delT;


    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
                /*Interior vertices*/

                /*Coeff  = [1,1,-4,1,1]*/
                /*Variables = [Ptop, Pleft, P, Pright and Pbtm] */
                P_colPtr[start] = thread_idx - Nx; 
                P_colPtr[start+1] = thread_idx - 1; 
                P_colPtr[start+2] = thread_idx ; 
                P_colPtr[start+3] = thread_idx+1 ; 
                P_colPtr[start+4] = thread_idx+Nx ; 


                P_val[start] = 1; 
                P_val[start+1] = 1; 
                P_val[start+2] = -4 ; 
                P_val[start+3] = 1 ; 
                P_val[start+4] = 1 ; 
                break;

            case '1':
                switch (thisV.BSide){
                    
                    case 'B' :
                    /*Bottom edge*/
                    /*Coeff  = [0,2,1,-4,1]*/
                    /*Var = [N/A, Ptop, Pleft, P, Pright]*/
                        P_colPtr[start] = thread_idx - (Ny-1)*Nx ; 
                        P_colPtr[start+1] = thread_idx - Nx; 
                        P_colPtr[start+2] = thread_idx - 1; 
                        P_colPtr[start+3] = thread_idx ; 
                        P_colPtr[start+4] = thread_idx+1 ; 
                       

                        P_val[start] = 0 ; 
                        P_val[start+1] = 2; 
                        P_val[start+2] = 1; 
                        P_val[start+3] = -4 ; 
                        P_val[start+4] = 1 ; 
                       
                    break;  

                    case 'T' :
                    /*Top edge*/
                    /*Coeff  = [1,-4,1,2,0]*/
                    /*Var = [Pleft, P, Pright, Pbtm, N/A]*/
                        P_colPtr[start] = thread_idx - 1; 
                        P_colPtr[start+1] = thread_idx; 
                        P_colPtr[start+2] = thread_idx + 1; 
                        P_colPtr[start+3] = thread_idx + Nx; 
                        P_colPtr[start+4] = thread_idx+Nx*(Ny-1) ; 
                       

                        P_val[start] = 1 ; 
                        P_val[start+1] = -4; 
                        P_val[start+2] = 1; 
                        P_val[start+3] = 2 ; 
                        P_val[start+4] = 0 ; 
                       
                    break;

                    case 'L' :
                    /*Left Edge*/
                        P_colPtr[start] = thread_idx - Nx; 
                        P_colPtr[start+1] = thread_idx; 
                        P_colPtr[start+2] = thread_idx + 1; 
                        P_colPtr[start+3] = thread_idx + Nx-1; 
                        P_colPtr[start+4] = thread_idx+Nx ; 
                       

                        P_val[start] = 1 ; 
                        P_val[start+1] = -4; 
                        P_val[start+2] = 2; 
                        P_val[start+3] = 0 ; 
                        P_val[start+4] = 1 ; 
                       
                    break;

                    case 'R' :
                    /*Right edge*/
                    /*Outlet - Values are hard-coded to 0 for convergence*/
                        P_colPtr[start] = thread_idx - Nx; 
                        P_colPtr[start+1] = thread_idx-Nx+1; 
                        P_colPtr[start+2] = thread_idx - 1; 
                        P_colPtr[start+3] = thread_idx ; 
                        P_colPtr[start+4] = thread_idx+Nx; 
                       

                        // P_val[start] = 1 ; 
                        // P_val[start+1] = 0; 
                        // P_val[start+2] = 2; 
                        // P_val[start+3] = -4 ; 
                        // P_val[start+4] = 1 ; 

                        P_val[start] = 0 ; 
                        P_val[start+1] = 0; 
                        P_val[start+2] = 0; 
                        P_val[start+3] = 1 ; 
                        P_val[start+4] = 0 ;                       

                    break;

                }

            case '2':
                switch (thisV.BSide){       
                    case 'W':
                    /*Top left*/

                    P_colPtr[start] = thread_idx; 
                    P_colPtr[start+1] = thread_idx+1; 
                    P_colPtr[start+2] = thread_idx+Nx-1; 
                    P_colPtr[start+3] = thread_idx + Nx ; 
                    P_colPtr[start+4] = thread_idx+Nx*(Ny-1); 


                    P_val[start] = -4 ; 
                    P_val[start+1] = 2; 
                    P_val[start+2] = 0; 
                    P_val[start+3] = 2 ; 
                    P_val[start+4] = 0 ; 
                    break;

                    case 'D':
                    /*Top right*/
                    P_colPtr[start] = thread_idx-Nx+1; 
                    P_colPtr[start+1] = thread_idx-1; 
                    P_colPtr[start+2] = thread_idx; 
                    P_colPtr[start+3] = thread_idx + Nx ; 
                    P_colPtr[start+4] = thread_idx+Nx*(Ny-1); 


                    P_val[start] = 0 ; 
                    P_val[start+1] = 2; 
                    P_val[start+2] = -4; 
                    P_val[start+3] = 2 ; 
                    P_val[start+4] = 0 ; 
                    break; 

                    case 'Z':
                    /*Bottom left*/

                    P_colPtr[start] = thread_idx-Nx*(Ny-1); 
                    P_colPtr[start+1] = thread_idx-Nx; 
                    P_colPtr[start+2] = thread_idx; 
                    P_colPtr[start+3] = thread_idx + 1 ; 
                    P_colPtr[start+4] = thread_idx+Nx-1; 


                    P_val[start] = 0 ; 
                    P_val[start+1] = 2; 
                    P_val[start+2] = -4; 
                    P_val[start+3] = 2 ; 
                    P_val[start+4] = 0 ; 
                    break;
                    case 'X':
                    /*Bottom Right*/

                    P_colPtr[start] = thread_idx-(Ny-1)*Nx; 
                    P_colPtr[start+1] = thread_idx-Nx; 
                    P_colPtr[start+2] = thread_idx-Nx+1; 
                    P_colPtr[start+3] = thread_idx -1 ; 
                    P_colPtr[start+4] = thread_idx; 


                    P_val[start] = 0 ; 
                    P_val[start+1] = 2; 
                    P_val[start+2] = 0; 
                    P_val[start+3] = 2 ; 
                    P_val[start+4] = -4 ; 
                    break; 
                }

        }

        /*Increment row pointer by 5, as each element has stencil of 5*/
        P_rowPtr[thread_idx + 1] = thread_idx*5 + 5;
    }

}


/*Calculates the RHS for the pressure-poisson equation*/
/*The RHS is simple the divergence of vlocity
*/

__global__
void update_PRHS(double* P_RHS, double* U, double* V,
                 const vertex* Domain, int rows, int Nx, int Ny,
                 double rho){



    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    vertex thisV = Domain[thread_idx]; 
    double delx = thisV.delx;
    double dely = thisV.dely; 

    double rightV, leftV, topV, btmV; 


    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
            /*Interior vertices*/
            rightV = U[thread_idx+1];
            leftV = U[thread_idx-1];
            topV = V[thread_idx-Nx]; 
            btmV = V[thread_idx+Nx];

            P_RHS[thread_idx] = (rightV - leftV)/(2*delx) + (topV - btmV)/(2*dely);
            break;

            case '1':
                switch (thisV.BSide){
                    case 'B' :
                        /*Bottom edge*/
                        rightV = U[thread_idx+1];
                        leftV = U[thread_idx-1];

                        if ((thisV.BType == '0') || (thisV.BType == '2')){
                        /*Inlet or wall*/
                        /*Assumed that V = (Vbtm + Vtop)/2. So Vbtm = 2*V - Vtop*/
                        
                        topV = V[thread_idx-Nx];                       
                        btmV =  2*thisV.VValue - topV;   
                    } else {
                        /*Symmetry - no gradient*/
                        topV = 0;                       
                        btmV =  0;  
                    }

                    P_RHS[thread_idx] = (rightV - leftV)/(2*delx) + (topV - btmV)/(4*dely);                       
                    break;  

                    case 'T' :
                        /*Top Edge*/
                        rightV = U[thread_idx+1];
                        leftV = U[thread_idx-1];

                        if ((thisV.BType == '0') || (thisV.BType == '2')){
                        /*Inlet or wall*/
                        /*Assumed that V = (Vbtm + Vtop)/2. So Vtop = 2*V - Vbtm*/
                        btmV = V[thread_idx+Nx];                       
                        topV =  2*thisV.VValue - btmV; 
                        } else {
                        topV = 0;                       
                        btmV =  0;  
                    }

                    P_RHS[thread_idx] = (rightV - leftV)/(2*delx) + (topV - btmV)/(4*dely);                       
                    break;  

                    case 'L' :
                        /*Left edge*/
                        topV = V[thread_idx-Nx];
                        btmV = V[thread_idx + Nx]; 

                        if ((thisV.BType == '0') || (thisV.BType == '2')){
                        /*Inlet or wall*/
                        rightV = U[thread_idx+1];                       
                        leftV =  2*thisV.UValue - rightV; 
                        } else {
                        rightV = 0;                       
                        leftV =  0;  
                        }
                    
                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(2*dely);                       
                    break;  

                    case 'R' :
                    /*Right edge*/
                        topV = V[thread_idx-Nx];
                        btmV = V[thread_idx + Nx]; 

                        if ((thisV.BType == '0') || (thisV.BType == '2')){
                        /*Inlet or wall*/

                        leftV = U[thread_idx-1];                       
                        rightV =  2*thisV.UValue - leftV; 
                        } else {
                        rightV = 0;                       
                        leftV =  0;  
                        }
                    
                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(2*dely);    
                       
                    break;

                }

            case '2':
                switch (thisV.BSide){       
                    case 'W':
                    /*Top left*/

                    btmV = V[thread_idx + Nx]; 
                    rightV = U[thread_idx + 1]; 
                    topV = 2*thisV.VValue -  btmV; 
                    leftV = 2*thisV.UValue -  rightV; 

                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(4*dely);    
                    break;

                    case 'D':
                    /*Top right*/
                    btmV = V[thread_idx + Nx]; 
                    leftV = U[thread_idx - 1]; 
                    topV = 2*thisV.VValue - btmV; 
                    rightV = 2*thisV.UValue -  leftV; 

                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(4*dely);    
                    break;

                    case 'Z':
                    /*Bottom left*/

                    topV = V[thread_idx - Nx]; 
                    rightV = U[thread_idx + 1]; 
                    btmV = 2*thisV.VValue -  topV; 
                    leftV = 2*thisV.UValue -  rightV; 

                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(4*dely);    
                    break;

                    case 'X':
                    /*Bottom Right*/

                    topV = V[thread_idx - Nx]; 
                    leftV = U[thread_idx - 1]; 
                    btmV = 2*thisV.VValue -  topV; 
                    rightV = 2*thisV.UValue - leftV; 

                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(4*dely);    
                    break;
 
                    break; 
                }

        }

    }

}


/*Final Step - Update the velocity with the pressure-corrected value
* U = U* + gradient of Pressure*/
__global__
void velPressureCorrection (double*P, double* U, double* V,const vertex* Domain, 
        int rows, int Nx, int Ny,double rho, double delT ){

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    vertex thisV = Domain[thread_idx]; 
    double delx = thisV.delx;
    double dely = thisV.dely; 
    double constTerm = delT/rho *delx*dely; 

    double right, left, top, btm; 

    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
            /*Interior Vertex*/
            right = P[thread_idx + 1];
            left = P[thread_idx - 1];
            top = P[thread_idx - Nx];
            btm = P[thread_idx + Nx];

            U[thread_idx] = U[thread_idx] - ( right - left)/(2*delx)*constTerm;
            V[thread_idx] = V[thread_idx] - (top-btm)/(2*dely)*constTerm;
            break; 

            case '1':
            /*Edge*/
            switch (thisV.BSide){

                case 'T':
                    /*Top Edge*/
                    right = P[thread_idx + 1];
                    left = P[thread_idx - 1];

                    if((thisV.VType == '0')||(thisV.VType == '1')){
                        top = 0;
                        btm = 0;
                    } else{
                        top = P[thread_idx-Nx];                       
                        btm =  2*thisV.PValue - top; 
                    }

                    U[thread_idx] = U[thread_idx] - ( right - left)/(2*delx)*constTerm;
                    V[thread_idx] = V[thread_idx] - (top-btm)/(4*dely)*constTerm;
                    break; 

                case 'B':
                    /*Btm Edge*/
                    right = P[thread_idx + 1];
                    left = P[thread_idx - 1];

                    if((thisV.VType == '0')||(thisV.VType == '1')){
                        top = 0;
                        btm = 0;
                    } else{
                        btm = P[thread_idx+Nx];                       
                        top =  2*thisV.PValue - btm; 
                    }

                    U[thread_idx] = U[thread_idx] - ( right - left)/(2*delx)*constTerm;
                    V[thread_idx] = V[thread_idx] - (top-btm)/(4*dely)*constTerm;
                    break;

                case 'R':
                    /*Right Edge*/
                    top = P[thread_idx - Nx];
                    btm = P[thread_idx + Nx];

                    if((thisV.VType == '0')||(thisV.VType == '1')){
                        right = 0;
                        left = 0;
                    } else{
                        left = P[thread_idx-1];                       
                        right =  2*thisV.PValue - left; 
                    } 

                    U[thread_idx] = U[thread_idx] - ( right - left)/(4*delx)*constTerm;
                    V[thread_idx] = V[thread_idx] - (top-btm)/(2*dely)*constTerm;
                    break;

                case 'L':
                    /*Left Edge*/
                    top = P[thread_idx - Nx];
                    btm = P[thread_idx + Nx];

                    if((thisV.VType == '0')||(thisV.VType == '1')){
                        right = 0;
                        left = 0;
                    } else{
                        right = P[thread_idx+1];                       
                        right =  2*thisV.PValue - right; 
                    } 

                    U[thread_idx] = U[thread_idx] - ( right - left)/(4*delx)*constTerm;
                    V[thread_idx] = V[thread_idx] - (top-btm)/(2*dely)*constTerm;
                    break;
            }


            break; 

            case '2':
                U[thread_idx] = U[thread_idx] ;
                V[thread_idx] = V[thread_idx] ;
                break; 

        }
    }
}


    
 
    
