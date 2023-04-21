/*
    The integrator is written in MKS HARM coordinates but takes in
    Kerr Schild input geodesic arrays. Most of the functions are
    borrowed from raptor written by Jordy Davelaar.
*/

#include "integrator.h"
#include "string.h"
#include "omp.h"

// Functions following this are in MKS
double _theta(double vartheta)
{
    return vartheta + 0.5 * h_ks * sin(2 * vartheta);
}

double delta(double M, double a, double r, double theta)
{
    double _delta = SQR(r) - 2 * M * r + SQR(a);
    return _delta;
}

double sigma(double M, double a, double r, double theta)
{
    double _sigma = SQR(r) + SQR(a * cos(theta));
    return _sigma;
}

// Covariant components of g_ij: only defined for j >= i
double g00l(double M, double a, double r, double theta)
{
    double _g00 = - (1 - 2 * M * r / sigma(M, a, r, theta));
    return _g00;
}

double g01l(double M, double a, double r, double theta)
{
    double _g01 = 2 * M * r / sigma(M, a, r, theta);
    return _g01;
}

double g02l(double M, double a, double r, double theta)
{
    return 0.;
}

double g03l(double M, double a, double r, double theta)
{
    double _g03 = - (2 * M * a * r / sigma(M, a, r, theta)) * SQR(sin(theta));
    return _g03;
}

double g11l(double M, double a, double r, double theta)
{
    double _g11 = 1 + 2 * M * r / sigma(M, a, r, theta);
    return _g11;
}

double g12l(double M, double a, double r, double theta)
{
    return 0.;
}

double g13l(double M, double a, double r, double theta)
{
    double _g13 = - (1 + 2 * M * r / sigma(M, a, r, theta)) * a * SQR(sin(theta));
    return _g13;
}

double g22l(double M, double a, double r, double theta)
{
    double _g22 = sigma(M, a, r, theta);
    return _g22;
}

double g23l(double M, double a, double r, double theta)
{
    return 0.;
}

double g33l(double M, double a, double r, double theta)
{
    double _g33 = (SQR(r) + SQR(a) + (2 * M * r * SQR(a) / sigma(M, a, r, theta)) * SQR(sin(theta))) * SQR(sin(theta));
    return _g33;
}

// Contravariant componetns of g^ij with j>=i
double g00u(double M, double a, double r, double theta)
{
    double _g00 = - (1 + 2 * M * r / sigma(M, a, r, theta));
    return _g00;
}

double g01u(double M, double a, double r, double theta)
{
    double _g01 = 2 * M * r / sigma(M, a, r, theta);
    return _g01;
}

double g02u(double M, double a, double r, double theta)
{
    return 0.;
}

double g03u(double M, double a, double r, double theta)
{
    return 0.;
}

double g11u(double M, double a, double r, double theta)
{
    double _g11 = delta(M, a, r, theta) / sigma(M, a, r, theta);
    return _g11;
}

double g12u(double M, double a, double r, double theta)
{
    return 0.;
}

double g13u(double M, double a, double r, double theta)
{
    double _g13 = a / sigma(M, a, r, theta);
    return _g13;
}

double g22u(double M, double a, double r, double theta)
{
    double _g22 = 1 / sigma(M, a, r, theta);
    return _g22;
}

double g23u(double M, double a, double r, double theta)
{
    return 0.;
}

double g33u(double M, double a, double r, double theta)
{
    double _g33 = 1 / (sigma(M, a, r, theta) * SQR(sin(theta)));
    return _g33;
}


void GetKSCovariantMetric(double g[4][4], double X[4])
{
    double r = X[1];
    double theta = X[2];
    double a = a_BH;
    double M = M_BH;
    g[0][0] = g00l(M, a, r, theta);
    g[0][1] = g01l(M, a, r, theta);
    g[0][2] = g02l(M, a, r, theta);
    g[0][3] = g03l(M, a, r, theta);
    g[1][0] = g[0][1];
    g[1][1] = g11l(M, a, r, theta);
    g[1][2] = g12l(M, a, r, theta);
    g[1][3] = g13l(M, a, r, theta);
    g[2][0] = g[0][2];
    g[2][1] = g[1][2];
    g[2][2] = g22l(M, a, r, theta);
    g[2][3] = g23l(M, a, r, theta);
    g[3][0] = g[0][3];
    g[3][1] = g[1][3];
    g[3][2] = g[2][3];
    g[3][3] = g33l(M, a, r, theta);
    return;
}

// Convert Kerr Schild to MKS
void ConvertCoords(double XKS[4], double XMKS[4])
{
    XMKS[0] = XKS[0];
    XMKS[1] = log(XKS[1] - R0_ks);
    XMKS[3] = XKS[3];
    // We can do a crappy root find
    // Want to get varphi such that theta = varphi + h_ks/2 sin(var_phi)
    // Limits of varphi: 0 - pi
    // Good thing is that the functions are roughly monotonic except for varphi=pi/2
    double low = 0., high=PI;
    double tol = 1.e-6;
    double mid = (low + high)/2;
    double theta = XKS[2], eval;
    eval = _theta(mid);
    int niter = 0;
    while(fabs(eval - theta) > tol)
    {
        if (eval < theta)
        {
            low = mid;
            mid = 0.5 * (low + high);
        }
        else if (eval > theta)
        {
            high = mid;
            mid = 0.5 * (low + high);
        }
        eval = _theta(mid);
        niter ++;
    }
    XMKS[2] = mid;
    return;
}

// Convert MKS to Kerr Schild
void ConvertCoordsInverse(double XMKS[4], double XKS[4])
{
    XKS[0] = XMKS[0];
    XKS[1] = R0_ks + exp(XMKS[1]);
    XKS[2] = XMKS[2] + (h_ks/2.) * sin(2 * XMKS[2]);
    XKS[3] = XMKS[3];
    return;
}

// Convert Kerr Schild vectors to MKS vectors
void ConvertVectors(double XMKS[4], double KKS[3], double KMKS[4])
{
    double dXKS_dXMKS[4];
    dXKS_dXMKS[0] = 1;
    dXKS_dXMKS[1] = exp(XMKS[1]);
    dXKS_dXMKS[2] = (1 + h_ks * cos(2 * XMKS[2]));
    dXKS_dXMKS[3] = 1.;

    for (int ii=0; ii<4; ii++)
    {
        KMKS[ii] = KKS[ii] / dXKS_dXMKS[ii];
    }
    return;
}

// Convert MKS vector to Kerr Schild vector
void ConvertVectorsInverse(double XMKS[4], double KMKS[4], double KKS[4])
{
    KKS[0] = KMKS[0];
    KKS[1] = KMKS[1] * exp(XMKS[1]);
    KKS[2] = KMKS[2] * (1 + h_ks * cos(2 * XMKS[2]));
    KKS[3] = KMKS[3];
    return;
}

// Return Covariant MKS metric
void GcovFunc(double XMKS[4], double g[4][4])
{
    double XKS[4];
    ConvertCoordsInverse(XMKS, XKS);
    GetKSCovariantMetric(g, XKS);
    double ds_dsks = exp(XMKS[1]);
    double dth_dthks = 1 + h_ks * cos(2 * XMKS[2]);

    g[0][1] *= ds_dsks;
    g[0][2] *= dth_dthks;
    g[1][0] *= ds_dsks;
    g[1][1] *= SQR(ds_dsks);
    g[1][2] *= ds_dsks * dth_dthks;
    g[1][3] *= ds_dsks;
    g[2][0] *= dth_dthks;
    g[2][1] *= ds_dsks * dth_dthks;
    g[2][2] *= SQR(dth_dthks);
    g[2][3] *= dth_dthks;
    g[3][1] *= ds_dsks;
    g[3][2] *= dth_dthks;
    return;
}

// Get MKS connection coefficients
void GetConnection(double X[4], double gamma[4][4][4])
{
    double r           = R0_ks + exp(X[1]);
    double r2          = r * r;
    double rprime      = exp(X[1]);
    double rprime2     = rprime * rprime;
    double rprimeprime = rprime;

    double theta = X[2] + 0.5 * h_ks * sin(2. * X[2]);
    double thetaprime = (1. +  h_ks * cos(2. * X[2]));
    double thetaprime2 = thetaprime * thetaprime;
    double thetaprimeprime = -2. * h_ks * sin(2. * X[2]);

    double costh       = cos(theta);
    double cos2th      = costh * costh;
    double sinth       = sin(theta);

    double sin2th      = sinth * sinth;
    double sintwoth    = sin(2. * theta);
    double cotth       = 1. / tan(theta);
    double a           = a_BH;
    double a2          = a * a;

    double Sigma       = r2 + a2 * cos2th;
    double Sigma2      = Sigma * Sigma;
    double Delta       = r2 - 2. * r + a2;
    double A           = Sigma * Delta + 2. * r * (r2 + a2);
    double Sigmabar    = 1. / Sigma;
    double Sigmabar2   = Sigmabar * Sigmabar;
    double Sigmabar3   = Sigmabar * Sigmabar * Sigmabar;
    double B           = (2. * r2 - Sigma) * Sigmabar3;
    double C           = r * Sigmabar - a2 * B * sin2th;

    // Gamma[t][mu][nu]
    gamma[0][0][0] = 2. * r * B;
    gamma[0][0][1] = (Sigma + 2. * r) * B * rprime;
    gamma[0][1][0] = gamma[0][0][1];
    gamma[0][0][2] = -a2 * r * sintwoth * Sigmabar2 * thetaprime;
    gamma[0][2][0] = gamma[0][0][2];
    gamma[0][0][3] = -2. * a * r * B * sin2th;
    gamma[0][3][0] = gamma[0][0][3];
    gamma[0][1][1] = 2. * (Sigma + r) * B * rprime2;
    gamma[0][1][2] = gamma[0][0][2] * rprime;
    gamma[0][2][1] = gamma[0][1][2];
    gamma[0][1][3] = -a * sin2th * gamma[0][0][1];
    gamma[0][3][1] = gamma[0][1][3];
    gamma[0][2][2] = -2. * r2 * Sigmabar * thetaprime2;
    gamma[0][2][3] = -a * sin2th * gamma[0][0][2];
    gamma[0][3][2] = gamma[0][2][3];
    gamma[0][3][3] = -2. * r * C * sin2th;

    // Gamma[r][mu][nu]
    gamma[1][0][0] = Delta * B / rprime;
    gamma[1][0][1] = (Delta - Sigma) * B;
    gamma[1][1][0] = gamma[1][0][1];
    gamma[1][2][0] = 0;
    gamma[1][0][2] = gamma[1][2][0];
    gamma[1][0][3] = -a * Delta * B * sin2th / rprime;
    gamma[1][3][0] = gamma[1][0][3];
    gamma[1][1][1] = rprimeprime / rprime - (2. * Sigma - Delta) * B * rprime;
    gamma[1][1][2] = -a2 * sinth * costh * Sigmabar * thetaprime;
    gamma[1][3][2] = 0;
    gamma[1][2][1] = gamma[1][1][2];
    gamma[1][2][3] = gamma[1][3][2];
    gamma[1][1][3] = a * (r * Sigmabar + (Sigma - Delta) * B) * sin2th;
    gamma[1][3][1] = gamma[1][1][3];
    gamma[1][2][2] = -r * Delta * Sigmabar /rprime * thetaprime2;
    gamma[1][3][3] = -Delta * C * sin2th / rprime;

    // Gamma[theta][mu][nu]
    gamma[2][0][0] = Sigmabar * gamma[0][0][2] / thetaprime2;
    gamma[2][0][1] = gamma[2][0][0] * rprime;
    gamma[2][1][0] = gamma[2][0][1];
    gamma[2][2][0] = 0;
    gamma[2][0][2] = gamma[2][2][0];
    gamma[2][0][3] = a * r * (r2 + a2) * sintwoth * Sigmabar3 / thetaprime;
    gamma[2][3][0] = gamma[2][0][3];
    gamma[2][1][1] = gamma[2][0][0] * rprime2;
    gamma[2][1][2] = r * Sigmabar * rprime;
    gamma[2][2][1] = gamma[2][1][2];
    gamma[2][1][3] = a * sinth * costh * (A + Sigma * (Sigma - Delta)) * Sigmabar3 * rprime / thetaprime;
    gamma[2][3][1] = gamma[2][1][3];
    gamma[2][3][2] = 0;
    gamma[2][2][3] = gamma[2][3][2];
    gamma[2][2][2] = thetaprimeprime / thetaprime + gamma[1][1][2];
    gamma[2][3][3] = -sinth * costh * (Delta * Sigma2 + 2. * r * (r2 + a2) * (r2 + a2)) * Sigmabar3 / thetaprime;

    // Gamma[phi][mu][nu]
    gamma[3][0][0] = a * B;
    gamma[3][0][1] = gamma[3][0][0] * rprime;
    gamma[3][1][0] = gamma[3][0][1];
    gamma[3][0][2] = -2. * a * r * cotth * Sigmabar2 * thetaprime;
    gamma[3][2][0] = gamma[3][0][2];
    gamma[3][0][3] = -a2 * B * sin2th;
    gamma[3][3][0] = gamma[3][0][3];
    gamma[3][1][1] = gamma[3][0][0] * rprime2;
    gamma[3][1][2] = -a * (Sigma + 2. * r) * cotth * Sigmabar2 * rprime * thetaprime;
    gamma[3][2][1] = gamma[3][1][2];
    gamma[3][1][3] = C * rprime;
    gamma[3][3][1] = gamma[3][1][3];
    gamma[3][2][2] = -a * r * Sigmabar * thetaprime2;
    gamma[3][2][3] = (cotth + a2 * r * sintwoth * Sigmabar2) * thetaprime;
    gamma[3][3][2] = gamma[3][2][3];
    gamma[3][3][3] = -a * C * sin2th;

    return;
}

// Get the adaptive stepsize for the geodesic
double stepsize(double X[4], double K[4], double epsilon)
{
    double dl, dlx1,dlxx1, dlx2, dlx3;
    double idlx1,idlxx1, idlx2, idlx3;

    double dlamb = epsilon;
    if (fabs(X[1] - R_max_MKS) < 2 * epsilon)
    {
        dlamb = EPS_END;
    }

    dlx1 = dlamb  / (fabs(K[1]) + SMALL);
    dlxx1 = dlamb * X[1]/ (fabs(K[1]) + SMALL);
    dlx2 = dlamb * X[2] / (fabs(K[2]) + SMALL);
    dlx3 = dlamb / (fabs(K[3]) + SMALL);
    idlx1 = 1. / (fabs(dlx1) + SMALL);
    idlxx1 = 1. / (fabs(dlxx1) + SMALL);
    idlx2 = 1. / (fabs(dlx2) + SMALL);
    idlx3 = 1. / (fabs(dlx3) + SMALL);
    if (X[1] > R_max_MKS * 1.005)
    {
        dl = 1/ (idlxx1 + idlx2 + idlx3);
    }
    else
    {
        dl = 1. / (idlx1 + idlx2 + idlx3);
    }
    return (dl);
}

// Initialize the photon wavevector derivative
void init_dKdlam(double X[], double Kcon[], double dK[])
{
  int k;
  double lconn[4][4][4];

  GetConnection(X, lconn);

  for (k = 0; k < 4; k++)
  {
    dK[k] =
        -2. * (Kcon[0] * (lconn[k][0][1] * Kcon[1] + lconn[k][0][2] * Kcon[2] +
                          lconn[k][0][3] * Kcon[3]) +
               Kcon[1] * (lconn[k][1][2] * Kcon[2] + lconn[k][1][3] * Kcon[3]) +
               lconn[k][2][3] * Kcon[2] * Kcon[3]);

    dK[k] -= (lconn[k][0][0] * Kcon[0] * Kcon[0] +
              lconn[k][1][1] * Kcon[1] * Kcon[1] +
              lconn[k][2][2] * Kcon[2] * Kcon[2] +
              lconn[k][3][3] * Kcon[3] * Kcon[3]);
  }
    return;
}

// Advance photon by a single timestep; assumes input coordinates to be MKS
void push_photon(double X[4], double Kcon[4], double dKcon[4],
                 double dl, int n)
{
    double lconn[4][4][4];
    double Kcont[4], K[4], dK;
    double Xcpy[4], Kcpy[4], dKcpy[4];
    double dl_2, err;
    int i, k, iter;
    
    if (X[1] < R_min_MKS)
        return;

    FAST_CPY(X, Xcpy);
    FAST_CPY(Kcon, Kcpy);
    FAST_CPY(dKcon, dKcpy);

    dl_2 = 0.5 * dl;
    
    /* Step the position and estimate new wave vector */
    for (i = 0; i < 4; i++) 
    {
        dK = dKcon[i] * dl_2;
        Kcon[i] += dK;
        K[i] = Kcon[i] + dK;
        X[i] += Kcon[i] * dl;
    }

    GetConnection(X, lconn);

    /* We're in a coordinate basis so take advantage of symmetry in the connection
    */
    iter = 0;
    do 
    {
        iter++;
        FAST_CPY(K, Kcont);

        err = 0.;
        for (k = 0; k < 4; k++) 
        {
            dKcon[k] =
                -2. *
                (Kcont[0] * (lconn[k][0][1] * Kcont[1] + lconn[k][0][2] * Kcont[2] +
                            lconn[k][0][3] * Kcont[3]) +
                Kcont[1] * (lconn[k][1][2] * Kcont[2] + lconn[k][1][3] * Kcont[3]) +
                lconn[k][2][3] * Kcont[2] * Kcont[3]);

            dKcon[k] -= (lconn[k][0][0] * Kcont[0] * Kcont[0] +
                        lconn[k][1][1] * Kcont[1] * Kcont[1] +
                        lconn[k][2][2] * Kcont[2] * Kcont[2] +
                        lconn[k][3][3] * Kcont[3] * Kcont[3]);

            K[k] = Kcon[k] + dl_2 * dKcon[k];
            err += fabs((Kcont[k] - K[k]) / (K[k] + SMALL));
        }

    } while (err > ETOL && iter < MAX_ITER);

    FAST_CPY(K, Kcon);

    // removing error in energy
    if (n < 7 && (err > ETOL || isnan(err) || isinf(err))) 
    {
        FAST_CPY(Xcpy, X);
        FAST_CPY(Kcpy, Kcon);
        FAST_CPY(dKcpy, dKcon);
        push_photon(X, Kcon, dKcon, 0.5 * dl, n + 1);
        push_photon(X, Kcon, dKcon, 0.5 * dl, n + 1);
    }

    /* done! */
    return;
}

// Function to test the geodesic integration: print out the energy, angular momentum and
// Carter's constant for a geodesic
// Everything must be in Kerr Schild
void TestGeodesic(double XKS[4], double KKS[4], double gcovKS[4][4], double* lambda, double *eta)
{
    // The first 8 elements are X^i and K^i
    double Kcov[4];

    for (int ii=0;ii<4;ii++)
    {
        Kcov[ii] = 0.;
        for (int jj=0; jj<4; jj++)
        {
            Kcov[ii] += KKS[jj] * gcovKS[ii][jj];
        }
    }

    //Add norm instead of eta
    double norm = KKS[0]*Kcov[0] + KKS[1]*Kcov[1] + KKS[2]*Kcov[2] + KKS[3]*Kcov[3];

    double p_t = Kcov[0];
    double p_theta = Kcov[2];
    double p_phi = Kcov[3];
    *lambda = -p_phi/p_t;
    double theta = XKS[2];
    *eta = norm;//SQR(p_theta/p_t) - SQR(a_BH * cos(theta)) + SQR(*lambda * cos(theta) / sin(theta));
    return;
}

// Initialize the geodesic to have the norm be 0 in the tetrad frame
int InitializeWavevector(double XKS[4], double KKS[4], double gcovKS[4][4])
{
    double A = gcovKS[0][0];
    double B = 0., C = 0.;

    for (int ii=1; ii<4; ii++)
    {
        B += 2. * KKS[ii] * gcovKS[ii][0];
        for (int jj=1; jj<4; jj++)
        {
            C += KKS[ii] * KKS[jj] * gcovKS[ii][jj];
        }
    }

    double sol1, sol2;
    sol1 = (- B + sqrt(B*B - 4*A*C)) / (2*A);
    sol2 = (- B - sqrt(B*B - 4*A*C)) / (2*A);

    if (sol1 > 0)
    {
        KKS[0] = sol1;
    }
    else if (sol2 > 0)
    {
        KKS[0] = sol2;
    }
    else
    {
        //printf("Cannot normalize the geodesic, aborting \n");
        return 0;
    }
    return 1;
}

// Save coordinates and conserved quantities in the output array
// The size of the output array must be 4 + 4 + 2 (X, K, C) where C are the conserved quantities
void kernel_GetGeodesic(double *Xinp, double *Kinp, int size, double epsilon, double *out)
{
    // Convert to MKS
    double X[4], K[4], gcovMKS[4][4];
    ConvertCoords(Xinp, X);
    ConvertVectors(X, Kinp, K);
    GcovFunc(X, gcovMKS);

    // Initialize the wavevector
    int success = InitializeWavevector(X, K, gcovMKS);
    // Bad geodesic - quit the
    if (success == 0)
    {
        for (int i=0; i<10; i++)
        {
            out[10 * size - i] = -1.;
        }
        return;
    }

    // Initalize the derivative
    double dKdlam[4];
    init_dKdlam(X, K, dKdlam);

    // Holder for new position
    double X_checkpoint;

    // Compute conserved quantitites
    double lambda, eta;
    double gcovKS[4][4];
    GetKSCovariantMetric(gcovKS, Xinp);
    TestGeodesic(Xinp, Kinp, gcovKS, &lambda, &eta);

    // Push the initial KS vector
    for (int ii=0; ii<4; ii++)
    {
        out[ii] = Xinp[ii];
        out[ii + 4] = Kinp[ii];
    }
    out[8] = lambda;
    out[9] = eta;

    X_checkpoint = X[1];

    int nsteps = 0, outsize = 1;
    while (X[1] < R_max_MKS)
    {
        nsteps++;
        double step = stepsize(X, K, epsilon);
        push_photon(X, K, dKdlam, step, 0);

        // exit if geodesic crosses inner edge or exceeds the number of steps
        if (X[1]<R_min_MKS || (nsteps>10000 && X[1] < 1.5*R_min_MKS))
        {
            //printf("Breaking on limit: nsteps=%d Radial MKS position=%f\n", nsteps, X[1]);
            break;
        }

        // Store result evenly at every r = 0.02 MKS
        if (fabs(X_checkpoint - X[1]) > 0.02 && outsize < size - 1)
        {
            X_checkpoint = X[1];
            ConvertCoordsInverse(X, Xinp);
            ConvertVectorsInverse(X, K, Kinp);
            GetKSCovariantMetric(gcovKS, Xinp);
            TestGeodesic(Xinp, Kinp, gcovKS, &lambda, &eta);
            for (int ii=0; ii<4; ii++)
            {
                out[10*outsize + ii] = Xinp[ii];
                out[10*outsize + 4 + ii] = Kinp[ii];
            }
            out[10*outsize + 8] = lambda;
            out[10*outsize + 9] = eta;
            outsize ++;
        }

    }

    // Record the last step at the array
    ConvertCoordsInverse(X, Xinp);
    ConvertVectorsInverse(X, K, Kinp);
    GetKSCovariantMetric(gcovKS, Xinp);
    TestGeodesic(Xinp, Kinp, gcovKS, &lambda, &eta);

    for (int ii=0; ii<4; ii++)
    {
        out[10 * size - 10 + ii] = Xinp[ii];
        out[10 * size - 10 + 4 + ii] = Kinp[ii];
    }
    out[10 * size - 2] = lambda;
    out[10 * size - 1] = eta;
    return;
}

// Call the above function in parallel
void kernel_GetGeodesicArray(double *Xinp, double *Kinp, int points_per_geodesic, int num_geodesics, 
                        double epsilon, double *out)
{
    const size_t size_subarr = 10 * points_per_geodesic;

    #pragma omp parallel for shared(out)
    for (int i=0; i<num_geodesics; i++)
    {
        double Xloc[4], Kloc[4];
        #pragma omp simd
        for (int jj=0; jj<4; jj++)
        {
            Xloc[jj] = Xinp[4 * i + jj];
            Kloc[jj] = Kinp[4 * i + jj];
        }
        double *outloc = malloc(size_subarr * sizeof(double));
        kernel_GetGeodesic(Xloc, Kloc, points_per_geodesic, epsilon, outloc);
        
        #pragma omp critical
        {
            #pragma omp simd
            for (size_t jj=0; jj<size_subarr; jj++)
            {
                out[i * size_subarr + jj] = outloc[jj];
            }
        }
        free(outloc);
    }

    return;
}
