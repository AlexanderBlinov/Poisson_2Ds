#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <string.h>

using namespace std;

double F(double x, double y, double z) {
    return 3 * exp(x + y + z);
}

double A0(double y, double z) {
    return exp(y + z);
}

double A1(double y, double z, double X) {
    return exp(X + y + z);
}

double B0(double x, double z) {
    return exp(x + z);
}

double B1(double x, double z, double Y) {
    return exp(x + Y + z);
}

double C0(double x, double y) {
    return exp(x + y);
}

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

    period[0] = true; period[1] = true;
    reorder = true;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &grid_comm);

    MPI_Cart_shift(grid_comm, 0, 1, &left, &right);
    MPI_Cart_shift(grid_comm, 1, 1, &up, &down);

    MPI_Cart_coords(grid_comm, rank, 2, coord);

    int rit = 200, tag = 31;
    double h1 = 0.005, h2 = 0.005, h3 = 0.005;
    double X = 1, Y = 1, Z = 1;
    double w = 1.7;

    int Nx = (int)(X / h1) - 1, Ny = (int)(Y / h2) - 1, Nz = (int)(Z / h3) - 1;

    int r2 = 20, r3 = 20, r4 = 3;
    int Q2 = (int)ceil((double)Nx / r2);
    int Q3 = (int)ceil((double)Ny / r3);
    int Q4 = (int)ceil((double)Nz / r4);

    int PQ2 = (int)ceil((double)Q2 / dim[0]);
    int PQ3 = (int)ceil((double)Q3 / dim[1]);

    double *U = (double *)calloc((size_t) PQ2 * PQ3 * r2 * r3 * r4 * Q4, sizeof(double));
    double *preBack = (double *)calloc((size_t)PQ2 * PQ3 * r2 * r4 * Q4, sizeof(double));
    double *preForth = (double *)calloc((size_t)PQ2 * PQ3 * r2 * r4 * Q4, sizeof(double));
    double *preLeft = (double *)calloc((size_t)PQ2 * PQ3 * r3 * r4 * Q4, sizeof(double));
    double *preRight = (double *)calloc((size_t)PQ2 * PQ3 * r3 * r4 * Q4, sizeof(double));

    MPI_Status status;
    MPI_Request request;

    MPI_Datatype uik_t;
    MPI_Type_vector(r2, r4, r3 * r4 * Q4, MPI_DOUBLE, &uik_t);
    MPI_Type_commit(&uik_t);

    MPI_Datatype uikq_t;
    MPI_Type_vector(r2, r4 * Q4, r3 * r4 * Q4, MPI_DOUBLE, &uikq_t);
    MPI_Type_commit(&uikq_t);

    MPI_Datatype ujk_t;
    MPI_Type_vector(r3, r4, r4 * Q4, MPI_DOUBLE, &ujk_t);
    MPI_Type_commit(&ujk_t);

    MPI_Datatype uik_t_full;
    MPI_Type_vector(r2, r4 * Q4, r3 * Q4 * r4, MPI_DOUBLE, &uik_t_full);
    MPI_Type_commit(&uik_t_full);
    MPI_Type_create_resized(uik_t_full, 0, r2 * r3 * r4 * Q4 * sizeof(double), &uik_t_full);
    MPI_Type_commit(&uik_t_full);

    MPI_Datatype ujk_t_full;
    MPI_Type_vector(PQ3, r3 * r4 * Q4, r2 * r3 * Q4 * r4, MPI_DOUBLE, &ujk_t_full);
    MPI_Type_commit(&ujk_t_full);
    MPI_Type_create_resized(ujk_t_full, 0, PQ3 * r2 * r3 * r4 * Q4 * sizeof(double), &ujk_t_full);
    MPI_Type_commit(&ujk_t_full);

    MPI_Datatype preik_t;
    MPI_Type_vector(r2, r4, r4 * Q4, MPI_DOUBLE, &preik_t);
    MPI_Type_commit(&preik_t);

    MPI_Datatype  prejk_t;
    MPI_Type_vector(r3, r4, r4 * Q4, MPI_DOUBLE, &prejk_t);
    MPI_Type_commit(&prejk_t);


    for (int i1 = 0; i1 < rit; ++i1) {
        if (i1 > 0) {
            if (rank != right) {
                MPI_Recv(preRight, PQ2 * PQ3 * r3 * r4 * Q4, MPI_DOUBLE, right, i1, grid_comm, &status);
            }
            if (rank != down) {
                MPI_Recv(preForth, PQ2 * PQ3 * r2 * r4 * Q4, MPI_DOUBLE, down, i1, grid_comm, &status);
            }
        }
        for (int igl2 = coord[0], iigl2 = 0; igl2 < Q2; igl2 += dim[0], ++iigl2) {
            for (int igl3 = coord[1], iigl3 = 0; igl3 < Q3; igl3 += dim[1], ++iigl3) {
                for (int igl4 = 0; igl4 < Q4; ++igl4) {
                    // RECV

                    if (((coord[0] == 0 && iigl2 > 0) || coord[0] > 0) && rank != left) {
                        MPI_Recv(preLeft + ((iigl2 * PQ3 + iigl3) * r3 * Q4 + igl4) * r4, 1, prejk_t, left,
                                 ((i1 * tag + iigl2) * tag + iigl3) + igl4, grid_comm, &status);
                    }

                    if (((coord[1] == 0 && iigl3 > 0) || coord[1] > 0) && rank != up) {
                        MPI_Recv(preBack + ((iigl2 * PQ3 + iigl3) * r2 * Q4 + igl4) * r4, 1, preik_t, up,
                                 ((i1 * tag + iigl2) * tag + iigl3) + igl4, grid_comm, &status);
                    }

                    for (int i2 = 0; i2 < min(r2, Nx - igl2 * r2); ++i2) {
                        for (int i3 = 0; i3 < min(r3, Ny - igl3 * r3); ++i3) {
                            for (int i4 = igl4 * r4; i4 < min((igl4 + 1) * r4, Nz); ++i4) {
                                int i = igl2 * r2 + i2, j = igl3 * r3 + i3, k = i4;
                                double uim, uip, ujm, ujp, ukm, ukp;
                                if (i == 0) {
                                    uim = A0((j + 1) * h2, (k + 1) * h3);
                                } else if (i2 == 0) {
                                    if (dim[0] == 1) {
                                        uim = U[((((iigl2 - 1) * PQ3 + iigl3 + 1) * r2 - 1) * r3 + i3) * r4 * Q4 + i4];
                                    } else {
                                        uim = preLeft[((iigl2 * PQ3 + iigl3) * r3 + i3) * r4 * Q4 + i4];
                                    }
                                } else {
                                    uim = U[(((iigl2 * PQ3 + iigl3) * r2 + i2 - 1) * r3 + i3) * r4 * Q4 + i4];
                                }

                                if (i == Nx - 1) {
                                    uip = A1((j + 1) * h2, (k + 1) * h3, X);
                                } else if (i2 == r2 - 1) {
                                    if (dim[0] == 1) {
                                        uip = U[(((iigl2 + 1) * PQ3 + iigl3) * r2 * r3 + i3) * r4 * Q4 + i4];
                                    } else {
                                        int x = (coord[0] == dim[0] - 1) ? iigl2 + 1 : iigl2;
                                        uip = preRight[((x * PQ3 + iigl3) * r3 + i3) * r4 * Q4 + i4];
                                    }
                                } else {
                                    uip = U[(((iigl2 * PQ3 + iigl3) * r2 + i2 + 1) * r3 + i3) * r4 * Q4 + i4];
                                }

                                if (j == 0) {
                                    ujm = B0((i + 1) * h1, (k + 1) * h3);
                                } else if (i3 == 0) {
                                    if (dim[1] == 1) {
                                        ujm = U[(((iigl2 * PQ3 + iigl3 - 1) * r2 + i2 + 1) * r3 - 1) * r4 * Q4 + i4];
                                    } else {
                                        ujm = preBack[((iigl2 * PQ3 + iigl3) * r2 + i2) * r4 * Q4 + i4];
                                    }
                                } else {
                                    ujm = U[(((iigl2 * PQ3 + iigl3) * r2 + i2) * r3 + i3 - 1) * r4 * Q4 + i4];
                                }

                                if (j == Ny - 1) {
                                    ujp = B1((i + 1) * h1, (k + 1) * h3, Y);
                                } else if (i3 == r3 - 1) {
                                    if (dim[1] == 1) {
                                        ujp = U[((iigl2 * PQ3 + iigl3 + 1) * r2 + i2) * r3 * r4 * Q4 + i4];
                                    } else {
                                        int x = (coord[1] == dim[1] - 1) ? iigl3 + 1 : iigl3;
                                        ujp = preForth[((iigl2 * PQ3 + x) * r2 + i2) * r4 * Q4 + i4];
                                    }
                                } else {
                                    ujp = U[(((iigl2 * PQ3 + iigl3) * r2 + i2) * r3 + i3 + 1) * r4 * Q4 + i4];
                                }

                                if (k == 0) {
                                    ukm = C0((i + 1) * h1, (j + 1) * h2);
                                } else {
                                    ukm = U[(((iigl2 * PQ3 + iigl3) * r2 + i2) * r3 + i3) * r4 * Q4 + i4 - 1];
                                }

                                if (k == Nz - 1) {
                                    ukp = C1((i + 1) * h1, (j + 1) * h2, Z);
                                } else {
                                    ukp = U[(((iigl2 * PQ3 + iigl3) * r2 + i2) * r3 + i3) * r4 * Q4 + i4 + 1];
                                }

                                double u = U[(((iigl2 * PQ3 + iigl3) * r2 + i2) * r3 + i3) * r4 * Q4 + i4];

                                U[(((iigl2 * PQ3 + iigl3) * r2 + i2) * r3 + i3) * r4 * Q4 + i4] =
                                        w * ((uip + uim) / (h1 * h1)
                                             + (ujp + ujm) / (h2 * h2)
                                             + (ukp + ukm) / (h3 * h3) - F((i + 1) * h1,
                                                                           (j + 1) * h2,
                                                                           (k + 1) * h3))
                                        / (2 / (h1 * h1) + 2 / (h2 * h2) + 2 / (h3 * h3))
                                        + (1 - w) * u;
                            }
                        }
                    }

                    // Send
                    if (((coord[0] == dim[0] - 1 && iigl2 < PQ2 - 1) || coord[0] < dim[0] - 1)
                        && rank != right && iigl2 * dim[0] + coord[0] + 1 < Q2) {
                        int sendTag = ((i1 * tag + iigl2) * tag + iigl3) + igl4;
                        if (coord[0] == dim[0] - 1) {
                            sendTag = ((i1 * tag + iigl2 + 1) * tag + iigl3) + igl4;
                        }
                        MPI_Isend(U + (((iigl2 * PQ3 + iigl3 + 1) * r2 - 1) * r3 * Q4 + igl4) * r4, 1, ujk_t, right,
                                  sendTag, grid_comm, &request);

                    }

                    if (((coord[1] == dim[1] - 1 && iigl3 < PQ3 - 1) || coord[1] < dim[1] - 1)
                        && rank != down && iigl3 * dim[1] + coord[1] + 1 < Q3) {
                        int sendTag = ((i1 * tag + iigl2) * tag + iigl3) + igl4;
                        if (coord[1] == dim[1] - 1) {

                            sendTag = ((i1 * tag + iigl2) * tag + iigl3 + 1) + igl4;

                        }
                        MPI_Isend(U + ((((iigl2 * PQ3 + iigl3) * r2 + 1) * r3 - 1) * Q4 + igl4) * r4, 1, uik_t, down,
                                  sendTag, grid_comm, &request);

                    }
                }
            }
        }
        if (i1 < rit - 1) {
            if (rank != left) {
                MPI_Isend(U, PQ2, ujk_t_full, left, i1 + 1, grid_comm, &request);
            }

            if (rank != up) {
                MPI_Isend(U, PQ2 * PQ3, uik_t_full, up, i1 + 1, grid_comm, &request);
            }

        }
    }

    double *R;
    if (rank == 0) {
        R = (double *)malloc(sizeof(double) * dim[0] * PQ2 * r2 * dim[1] * PQ3 * r3 * r4 * Q4);

        for (int i2 = 0; i2 < PQ2; ++i2) {
            for (int i3 = 0; i3 < PQ3; ++i3) {
                for (int i = 0; i < r2; ++i) {
                    memcpy(R + ((i2 * dim[0] * r2 + i) * PQ3 + i3) * dim[1] * r3 * r4 * Q4,
                           U + ((i2 * PQ3 + i3) * r2 + i) * r3 * r4 * Q4,
                           r3 * r4 * Q4 * sizeof(double));
                }
            }
        }

        int sender, self[2], coords[2];
        MPI_Cart_coords(grid_comm, rank, 2, self);
        for (int n = 0; n < dim[0]; ++n) {
            for (int m = 0; m < dim[1]; ++m) {
                if (n != self[0] || m != self[1]) {
                    coords[0] = n; coords[1] = m;
                    MPI_Cart_rank(grid_comm, coords, &sender);
                    MPI_Recv(U, PQ2 * PQ3 * r2 * r3 * r4 * Q4, MPI_DOUBLE, sender,
                             10005000, grid_comm, &status);
                }

                for (int i2 = 0; i2 < PQ2; ++i2) {
                    for (int i3 = 0; i3 < PQ3; ++i3) {
                        for (int i = 0; i < r2; ++i) {
                            memcpy(R + ((((i2 * dim[0] + n) * r2 + i) * PQ3 + i3) * dim[1] + m) * r3 * r4 * Q4,
                                   U + ((i2 * PQ3 + i3) * r2 + i) * r3 * r4 * Q4,
                                   r3 * r4 * Q4 * sizeof(double));
                        }
                    }
                }
            }
        }
    } else {
        MPI_Send(U, PQ2 * PQ3 * r2 * r3 * r4 * Q4, MPI_DOUBLE, 0, 10005000, grid_comm);
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
                    fprintf(f, "%f ", R[(i * dim[1] * PQ3 * r3 + j) * r4 * Q4 + k]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "-----------------------------------------------\n");
        }
        fclose(f);
    }

    return 0;
}