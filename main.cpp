#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

using namespace std;


// Neodnorodnost'
double F(double x, double y, double z) {
    return 3 * exp(x + y + z);
}

// Granichnoe uslovie pri x=0
double A0(double y, double z) {
    return exp(y + z);
}

// Granichnoe uslovie pri x=X
double A1(double y, double z, double X) {
    return exp(X + y + z);
}

// Granichnoe uslovie pri y=0
double B0(double x, double z) {
    return exp(x + z);
}

// Granichnoe uslovie pri y=Y
double B1(double x, double z, double Y) {
    return exp(x + Y + z);
}

// Granichnoe uslovie pri z=0
double C0(double x, double y) {
    return exp(x + y);
}

// Granichnoe uslovie pri z=Z
double C1(double x, double y, double Z) {
    return exp(x + y + Z);
}

int main(int argc, char *argv[]) {
    int rank;
    int dim[2], period[2], reorder, coord[2];
    dim[0] = atoi(argv[1]); dim[1] = atoi(argv[2]);
    int up, down, right, left;
    double start, end;

    MPI_Comm grid_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    period[0] = false; period[1] = false;
    reorder = true;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &grid_comm);

    // Nahozhedie sosednih processov
    MPI_Cart_shift(grid_comm, 0, 1, &left, &right);
    MPI_Cart_shift(grid_comm, 1, 1, &up, &down);


    // Vichislenie osnovnih parametrov oblasti reshenija
    int rit = 300, tag = 1000;
    double h1 = 0.01, h2 = 0.01, h3 = 0.01;
    double X = 1, Y = 1, Z = 1;
    double w = 1.7;

    int Nx = (int)(X / h1) - 1, Ny = (int)(Y / h2) - 1, Nz = (int)(Z / h3) - 1;

    int r4 = 20;
    int Q4 = (int)ceil((double)Nz / r4);

    int Q2 = dim[0], Q3 = dim[1];
    int r2 = (int)ceil((double)Nx / Q2);
    int r3 = (int)ceil((double)Ny / Q3);

    MPI_Cart_coords(grid_comm, rank, 2, coord);
    int igl2 = coord[0];
    int igl3 = coord[1];

    double *U = (double *)calloc((size_t)r2 * r3 * r4 * Q4, sizeof(double));
    double *preBack = (double *)calloc((size_t)r2 * r4 * Q4, sizeof(double));
    double *preForth = (double *)calloc((size_t)r2 * r4 * Q4, sizeof(double));
    double *preLeft = (double *)calloc((size_t)r3 * r4 * Q4, sizeof(double));
    double *preRight = (double *)calloc((size_t)r3 * r4 * Q4, sizeof(double));

    MPI_Status status;

    MPI_Datatype uik_t;
    MPI_Type_vector(r2, r4, r3 * r4 * Q4, MPI_DOUBLE, &uik_t);
    MPI_Type_commit(&uik_t);

    MPI_Datatype ujk_t;
    MPI_Type_vector(r3, r4, r4 * Q4, MPI_DOUBLE, &ujk_t);
    MPI_Type_commit(&ujk_t);

    MPI_Datatype uik_t_full;
    MPI_Type_vector(r2, r4 * Q4, r3 * Q4 * r4, MPI_DOUBLE, &uik_t_full);
    MPI_Type_commit(&uik_t_full);

    MPI_Datatype preik_t;
    MPI_Type_vector(r2, r4, r4 * Q4, MPI_DOUBLE, &preik_t);
    MPI_Type_commit(&preik_t);

    MPI_Datatype  prejk_t;
    MPI_Type_vector(r3, r4, r4 * Q4, MPI_DOUBLE, &prejk_t);
    MPI_Type_commit(&prejk_t);

    for (int i1 = 0; i1 < rit; ++i1) {
        if (i1 > 0) {
            if (right != -1) {
//                printf("%d (%d) recvs right from %d\n", rank, i1, right);
                MPI_Recv(preRight, r3 * r4 * Q4, MPI_DOUBLE, right, i1, grid_comm, &status);
//                printf("%d (%d) recved right from %d\n", rank, i1, right);
            }
            if (down != -1) {
//                printf("%d (%d) recvs down from %d\n", rank, i1, down);
                MPI_Recv(preForth, r2 * r4 * Q4, MPI_DOUBLE, down, i1, grid_comm, &status);
//                printf("%d (%d) recved down from %d\n", rank, i1, down);
            }
        }
        for (int igl4 = 0; igl4 < Q4; ++igl4) {
            if (left != -1) {
//                printf("%d (%d, %d) recvs left from %d\n", rank, i1, igl4, left);
                MPI_Recv(preLeft + igl4 * r4, 1, prejk_t, left, i1 * tag + igl4, grid_comm, &status);
//                printf("%d (%d, %d) recved left from %d\n", rank, i1, igl4, left);
            }

            if (up != -1) {
//                printf("%d (%d, %d) recvs up form %d\n", rank, i1, igl4, up);
                MPI_Recv(preBack + igl4 * r4, 1, preik_t, up, i1 * tag + igl4, grid_comm, &status);
//                printf("%d (%d, %d) recved up from %d\n", rank, i1, igl4, up);
            }

            for (int i2 = 0; i2 < min(r2, Nx - igl2 * r2); ++i2) {
                for (int i3 = 0; i3 < min(r3, Ny - igl3 * r3); ++i3) {
                    for (int i4 = igl4 * r4; i4 < min((igl4 + 1) * r4, Nz); ++i4) {
                        int i = igl2 * r2 + i2, j = igl3 * r3 + i3, k = i4;
                        double uim, uip, ujm, ujp, ukm, ukp;
                        if (i == 0) {
                            uim = A0((j + 1) * h2, (k + 1) * h3);
                        } else if (i2 == 0) {
                            uim = preLeft[i3 * r4 * Q4 + i4];
                        } else {
                            uim = U[((i2 - 1) * r3 + i3) * r4 * Q4 + i4];
                        }

                        if (i == Nx - 1) {
                            uip = A1((j + 1) * h2, (k + 1) * h3, X);
                        } else if (i2 == r2 - 1) {
                            uip = preRight[i3 * r4 * Q4 + i4];
                        } else {
                            uip = U[((i2 + 1) * r3 + i3) * r4 * Q4 + i4];
                        }

                        if (j == 0) {
                            ujm = B0((i + 1) * h1, (k + 1) * h3);
                        } else if (i3 == 0) {
                            ujm = preBack[i2 * r4 * Q4 + i4];
                        } else {
                            ujm = U[(i2 * r3 + i3 - 1) * r4 * Q4 + i4];
                        }

                        if (j == Ny - 1) {
                            ujp = B1((i + 1) * h1, (k + 1) * h3, Y);
                        } else if (i3 == r3 - 1) {
                            ujp = preForth[i2 * r4 * Q4 + i4];
                        } else {
                            ujp = U[(i2 * r3 + i3 + 1) * r4 * Q4 + i4];
                        }

                        if (k == 0) {
                            ukm = C0((i + 1) * h1, (j + 1) * h2);
                        } else {
                            ukm = U[(i2 * r3 + i3) * r4 * Q4 + i4 - 1];
                        }

                        if (k == Nz - 1) {
                            ukp = C1((i + 1) * h1, (j + 1) * h2, Z);
                        } else {
                            ukp = U[(i2 * r3 + i3) * r4 * Q4 + i4 + 1];
                        }

                        double u = U[(i2 * r3 + i3) * r4 * Q4 + i4];

                        U[(i2 * r3 + i3) * r4 * Q4 + i4] = w * ((uip + uim) / (h1 * h1)
                                                                + (ujp + ujm) / (h2 * h2)
                                                                + (ukp + ukm) / (h3 * h3) - F((i + 1) * h1,
                                                                                              (j + 1) * h2,
                                                                                              (k + 1) * h3))
                                                           / (2 / (h1 * h1) + 2 / (h2 * h2) + 2 / (h3 * h3))
                                                           + (1 - w) * u;
                    }
                }
            }
            if (right != -1) {
//                printf("%d (%d, %d) sends right to %d\n", rank, i1, igl4, right);
                MPI_Send(U + (r2 - 1) * r3 * r4 * Q4 + igl4 * r4, 1, ujk_t, right, i1 * tag + igl4, grid_comm);
//                printf("%d (%d, %d) sent right to %d\n", rank, i1, igl4, right);
            }

            if (down != -1) {
//                printf("%d (%d, %d) sends down to %d\n", rank, i1, igl4, down);
                MPI_Send(U + (r3 - 1) * r4 * Q4 + igl4 * r4, 1, uik_t, down, i1 * tag + igl4, grid_comm);
//                printf("%d (%d, %d) sent down to %d\n", rank, i1, igl4, down);
            }
        }
        if (i1 < rit - 1) {
            if (left != -1) {
//                printf("%d (%d) sends left to %d\n", rank, i1, left);
                MPI_Send(U, r3 * r4 * Q4, MPI_DOUBLE, left, i1 + 1, grid_comm);
//                printf("%d (%d) sent left to %d\n", rank, i1, left);
            }
            if (up != -1) {
//                printf("%d (%d) sends up to %d\n", rank, i1, up);
                MPI_Send(U, 1, uik_t_full, up, i1 + 1, grid_comm);
//                printf("%d (%d) sent up to %d\n", rank, i1, up);
            }
        }
    }

    double *R;
    if (rank == 0) {
        R = (double *)malloc(sizeof(double) * r2 * Q2 * r3 * Q3 * r4 * Q4);

        MPI_Datatype r_t;
        MPI_Type_vector(r2, r3 * r4 * Q4, r3 * Q3 * r4 * Q4, MPI_DOUBLE, &r_t);
        MPI_Type_commit(&r_t);

        int self[2];
        MPI_Cart_coords(grid_comm, rank, 2, self);
        for (int n = 0; n < dim[0]; ++n) {
            for (int m = 0; m < dim[1]; ++m) {
                if (n == self[0] && m == self[1]) {
                    for (int i = 0; i < r2; ++i) {
                        memcpy(R + ((r2 * n + i) * Q3 + m) * r3 * r4 * Q4,
                               U + i * r3 * r4 * Q4,
                               r3 * r4 * Q4 * sizeof(double));
                    }
                    continue;
                }

                int sender;
                int coords[2];
                coords[0] = n; coords[1] = m;
                MPI_Cart_rank(grid_comm, coords, &sender);
                MPI_Recv(R + (n * r2 * Q3 + m) * r3 * r4 * Q4, 1, r_t, sender, 10005000, grid_comm, &status);
            }
        }
    } else {
        MPI_Send(U, r2 * r3 * r4 * Q4, MPI_DOUBLE, 0, 10005000, grid_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    MPI_Finalize();

    if (rank == 0) {
        printf("Time: %f\n", end - start);

        FILE *f = fopen("output.txt", "w");
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    fprintf(f, "%f ", R[(i * r3 * Q3 + j) * r4 * Q4 + k]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "-----------------------------------------------\n");
        }
        fclose(f);
    }

    return 0;
}