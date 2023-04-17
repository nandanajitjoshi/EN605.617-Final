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

/*Data type for storing information of a mesh node*/
struct vertex {
    char VType;     // 0 - interior, 1- boundary edge, 2- corner point

    char BSide;     //  R - right, L - left, T - top, B- bottom for edges...
                    //  W- top left, D - top rt, Z - btm lft, X - btm rt for corners

    char BType;     //  Type of boundary condition applied ....
                    //  0- Inlet/Outlet (specified value), 1- Wall, 2- Symmetry

    double dely;    // Size of vertex in y (vertical) direction
    double delx;    // Size of vertex in x (horizontal) direction
    double UValue;  //If predefined value for U,V, P is set for the vertex
    double VValue;  // 0 by default. Values are specified only at boundaries
    double PValue; 
}; 

#define U0 1


/*Creates a 2D Ny*Nx array of vertexes */
/*The edges of the array are defined as boundaries*/
/*The vertices are arranged in a column major order*/
/*input: Lx, Ly denotes the length of the domain
* input : Nx,Ny denote the number of vertexes in x and y direction
* output : Domain is the actual 2D array
*
* Preset BC values are used at present, will be modified to take
* user inputs in the future
*/
__global__
void Mesh (int Lx, int Ly, int Nx,int Ny, vertex* Domain){

    int rows = Ny*Nx; 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < rows){
        if (i == 0){
            /* Top left corner*/
            Domain[i]. VType = '2';
            Domain[i].BSide = 'W';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/(2*Nx);
            Domain[i].BType = '0';
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].PValue = 0;

        } else if (i== (Ny-1)){
            /*Top Right Corner*/
            Domain[i]. VType = '2';
            Domain[i].BSide = 'D';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/(2*Nx);
            Domain[i].BType = '0';
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].PValue = 0;

        } else if (i==(Ny*(Nx-1))){
            /*Bottom Left Corner*/
            Domain[i]. VType = '2';
            Domain[i].BSide = 'Z';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/(2*Nx);
            Domain[i].BType = '0';
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].PValue = 0;

        } else if (i== (rows -1)){
            /*Bottom Right Corner*/
            Domain[i]. VType = '2';
            Domain[i].BSide = 'X';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/(2*Nx);
            Domain[i].BType = '0';
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].PValue = 0;

        } else if  (i < Ny){
            /*Top Edge*/
            Domain[i]. VType = '1';
            Domain[i].BSide = 'T';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/Nx;
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].BType = '1';  //Symmetry BC
            Domain[i].PValue = 0;

        } else if (i%Ny == 0){
            /*Left Edge*/
            Domain[i]. VType = '1';
            Domain[i].BSide = 'L';
            Domain[i]. dely = (double)Ly/Ny;
            Domain[i].delx = (double) Lx/(2*Nx);
            Domain[i].UValue = U0;
            Domain[i].VValue = 0;
            Domain[i].BType = '0';  //Inlet BC
            Domain[i].PValue = 0;

        } else if (i%Ny == (Ny-1)){
            /*Right Edge*/
            Domain[i]. VType = '1';
            Domain[i].BSide = 'R';
            Domain[i]. dely = (double)Ly/Ny;
            Domain[i].delx = (double)Lx/(2*Nx);
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].BType = '0';   //Fixed value (Outlet BC)
            Domain[i].PValue = 0;

        } else if (i >= (Ny*(Nx-1)) ){
            /*Bottom Edge*/
            Domain[i]. VType = '1';
            Domain[i].BSide = 'B';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/Nx;
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].BType = '1';  //Symmetry BC
            Domain[i].PValue = 0;

        } else {
            /*Internal Points*/
            Domain[i]. VType = '0';
            Domain[i].BSide = '0';
            Domain[i].dely = (double)Ly/Ny;
            Domain[i].delx = (double)Lx/Nx;
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].BType = '0';
            Domain[i].PValue = 0;
        } 
    }


      
}