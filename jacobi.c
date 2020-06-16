#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

// Boundary value at the top of the domain
#define TOP_VALUES 1.0
// Boundary value at the bottom of the domain
#define BOTTOM_VALUES 10.0
// The maximum number of iterations
#define MAX_ITERATIONS 5000000
// The convergence to terminate at
#define CONVERGENCE_ACCURACY 1e-4
// How often to report the norm
#define REPORT_NORM_PERIOD 1000
// How often to compute the norm
#define COMPUTE_NORM_PERIOD 10

int nx, ny;

void initialise(double*, double*, int, int, int);

int main(int argc, char * argv[]) {
  int size, myrank;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 3) {
    if (myrank==0) fprintf(stderr, "You must provide the size in X and size in Y as arguments to this code\n");
    return -1;
  }
  nx=atoi(argv[1]);
  ny=atoi(argv[2]);

  if (myrank==0) printf("Solving to accuracy of %.0e, global system size is x=%d y=%d\n", CONVERGENCE_ACCURACY, nx, ny);
  int local_nx=nx/size;
  if (local_nx * size < nx) {
    if (myrank < nx - local_nx * size) local_nx++;
  }

  double * u_k = malloc(sizeof(double) * (local_nx + 2) * ny);
  double * u_kp1 = malloc(sizeof(double) * (local_nx + 2) * ny);
  double start_time;

  initialise(u_k, u_kp1, local_nx, myrank, size);

  double rnorm=0.0, bnorm=0.0, norm, tmpnorm=0.0;
  MPI_Request requests[]={MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  int i,j,k;
  for (i=1;i<=local_nx;i++) {
    for (j=0;j<ny;j++) {
      tmpnorm=tmpnorm+pow(u_k[j+(i*ny)]*4-u_k[(j-1) + (i*ny)]-u_k[(j+1) + (i*ny)] - u_k[j+((i-1)*ny)] - u_k[j+((i+1)*ny)], 2);
    }
  }

  MPI_Allreduce(&tmpnorm, &bnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  bnorm=sqrt(bnorm);

  start_time=MPI_Wtime();

  if (myrank > 0) {
    MPI_Isend(&u_k[ny], ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&u_k[0], ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, &requests[1]);
  }
  if (myrank < size-1) {
    MPI_Isend(&u_k[local_nx * ny], ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(&u_k[(local_nx+1) * ny], ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, &requests[3]);
  }
  MPI_Request requestColl;

  for (k=0;k<MAX_ITERATIONS;k++) {

    if (k % COMPUTE_NORM_PERIOD == 0) {
      tmpnorm=0.0;
      for (i=1;i<=local_nx;i++) {
        for (j=0;j<ny;j++) {
          tmpnorm=tmpnorm+pow(u_k[j+(i*ny)]*4-u_k[(j-1) + (i*ny)]-u_k[(j+1) + (i*ny)] - u_k[j+((i-1)*ny)] - u_k[j+((i+1)*ny)], 2);
        }
      }
      MPI_Iallreduce(&tmpnorm, &rnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &requestColl);
    }

    MPI_Wait(&requests[1], MPI_STATUS_IGNORE); // recv UP
    i=1; {
      for (j=0;j<ny;j++) {
        u_kp1[j+(i*ny)]=0.25 * (u_k[(j-1) + (i*ny)]+u_k[(j+1) + (i*ny)] + u_k[j+ ((i+1)*ny)] + u_k[j+ ((i-1)*ny)]);
      }
    }
    if (myrank > 0)
      MPI_Irecv(&u_k[0], ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, &requests[1]); // recv UP

    MPI_Wait(&requests[3], MPI_STATUS_IGNORE); // recv DOWN
    i=local_nx; {
      for (j=0;j<ny;j++) {
        u_kp1[j+(i*ny)]=0.25 * (u_k[(j-1) + (i*ny)]+u_k[(j+1) + (i*ny)] + u_k[j+ ((i+1)*ny)] + u_k[j+ ((i-1)*ny)]);
      }
    }
    if (myrank < size-1)
      MPI_Irecv(&u_k[(local_nx+1) * ny], ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, &requests[3]); // recv DOWN

    for (i=2;i<=local_nx-1;i++) {
      for (j=0;j<ny;j++) {
        u_kp1[j+(i*ny)]=0.25 * (u_k[(j-1) + (i*ny)]+u_k[(j+1) + (i*ny)] + u_k[j+ ((i+1)*ny)] + u_k[j+ ((i-1)*ny)]);
      }
    }

    if (k % COMPUTE_NORM_PERIOD == 0) {
      MPI_Wait(&requestColl, MPI_STATUS_IGNORE);
      norm=sqrt(rnorm)/bnorm;
      if (norm < CONVERGENCE_ACCURACY) break;
    }
    
    MPI_Wait(&requests[0], MPI_STATUS_IGNORE); // send UP
    memcpy(&u_k[           ny], &u_kp1[           ny], sizeof(double) * ny);
    if (myrank > 0)
      MPI_Isend(&u_k[ny], ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, &requests[0]); // send UP

    MPI_Wait(&requests[2], MPI_STATUS_IGNORE); // send DOWN
    memcpy(&u_k[local_nx * ny], &u_kp1[local_nx * ny], sizeof(double) * ny);
    if (myrank < size-1)
      MPI_Isend(&u_k[local_nx * ny], ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, &requests[2]); // send DOWN

    memcpy(&u_k[       2 * ny], &u_kp1[       2 * ny], sizeof(double) * ny * (local_nx-2));

    if (k % REPORT_NORM_PERIOD == 0 && myrank==0) printf("Iteration= %d Relative Norm=%e\n", k, norm);
  }

  MPI_Wait(&requests[0], MPI_STATUS_IGNORE); // send UP
  MPI_Wait(&requests[1], MPI_STATUS_IGNORE); // recv UP
  MPI_Wait(&requests[2], MPI_STATUS_IGNORE); // send DOWN
  MPI_Wait(&requests[3], MPI_STATUS_IGNORE); // recv DOWN

  if (myrank==0) printf("\nTerminated on %d iterations, Relative Norm=%e, Total time=%e seconds\n", k, norm,
                        MPI_Wtime() - start_time);
  free(u_k);
  free(u_kp1);

  MPI_Finalize();
  return 0;
}




/**
 * Initialises the arrays, such that u_k contains the boundary conditions at the start and end points and all other
 * points are zero. u_kp1 is set to equal u_k
 */
void initialise(double * u_k, double * u_kp1, int local_nx, int myrank, int size) {
  int i, j;
  for (j=0;j<ny;j++) {
    u_kp1[j]=u_k[j]=myrank==0 ? TOP_VALUES: 0;
  }
  for (j=0;j<ny;j++) {
    u_kp1[j+(ny*(local_nx+1))]=u_k[j+(ny*(local_nx+1))]=myrank==size-1? BOTTOM_VALUES: 0;
  }
  for (i=1;i<=local_nx;i++) {
    for (j=0;j<ny;j++) {
      u_kp1[j+(ny*i)]=u_k[j+(ny*i)]=0;
    }
  }
}
