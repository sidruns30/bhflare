#include "HAMR_MHD.hpp"

/*
    Similar to BHAC MHD but compute fluid quantities for HAMR data
    Analysis is done in regular KS coordinates
*/

void HAMR_MHD::GetTemp(ARRAY &temp, const ARRAY2D &COORDS, const ARRAY2D &PRIMS)
{
    print("Computing temp");
    size_t i, N = COORDS[0].size();
    temp.clear();

    #pragma omp parallel
    {
        ARRAY temp_private;
        #pragma omp for nowait schedule(static) private(i, N)
        for (i=0; i<N; i++)
        {
            double _rho = PRIMS[iRHO][i], _P = PRIMS[iP][i];
            temp_private.push_back(_P / _rho);
        }
        #pragma omp for schedule(static) ordered
        for(int i=0; i<omp_get_num_threads(); i++)
        {
            #pragma omp ordered
            temp.insert(temp.end(), temp_private.begin(), temp_private.end());
        }
    }
    return;
}

void HAMR_MHD::GetBsqr(ARRAY &Bsqr, const ARRAY2D &COORDS, const ARRAY2D &PRIMS)
{
    print("Computing bsqr");
    size_t i, N = COORDS[0].size();
    Bsqr.clear();

    #pragma omp parallel
    {
        ARRAY Bsqr_private;
        #pragma omp for nowait schedule(static) private(i, N)
        for (i=0; i<N; i++)
        {
            double x = COORDS[0][i], y = COORDS[1][i], z = COORDS[2][i];

            // Get KS coordinates
            double XKS[NDIM];
            CartToKS(x, y, z, XKS);
            
            // Now get the arrays in KS
            double Bi_con[NDIM] = {PRIMS[iB0][i], PRIMS[iB1][i], PRIMS[iB2][i], PRIMS[iB3][i]};
            double Bi_cov[NDIM];

            UpperToLower(2, XKS, Bi_con, Bi_cov);

            double sum = 0.;
            #pragma omp simd reduction(+:sum)
            for (size_t k=0; k<NDIM; k++)
            {
                sum += Bi_con[k] * Bi_cov[k];
            }
            Bsqr_private.push_back(sum);
        }
        #pragma omp for schedule(static) ordered
        for(int i=0; i<omp_get_num_threads(); i++)
        {
            #pragma omp ordered
            Bsqr.insert(Bsqr.end(), Bsqr_private.begin(), Bsqr_private.end());
        }
    }
    return;
}

void HAMR_MHD::GetSigma(ARRAY &sigma, ARRAY &b2, const ARRAY2D &COORDS, const ARRAY2D &PRIMS)
{
    print("Computing sigma");
    size_t i, N;
    N = COORDS[0].size();

    sigma.clear();

    // Check if Bsqr is computed
    if (b2.size() != N)
    {
        throw std::invalid_argument("bsq must be computed \n");
    }

    #pragma omp parallel
    {
        ARRAY sigma_private;
        #pragma omp for nowait schedule(static) private(i, N)
        for (i=0; i<N; i++)
        {
            double _sigma = b2[i] / PRIMS[iRHO][i];
            sigma_private.push_back(_sigma);
        }
        #pragma omp for schedule(static) ordered
        for(int i=0; i<omp_get_num_threads(); i++)
        {
            #pragma omp ordered
            sigma.insert(sigma.end(), sigma_private.begin(), sigma_private.end());
        }
    }
    return;
}

void HAMR_MHD::GetBeta(ARRAY &beta, ARRAY &b2, const ARRAY2D &COORDS, const ARRAY2D &PRIMS)
{
    print("Computing beta");
    size_t i, N;
    N = COORDS[0].size();

    beta.clear();
    
    // Check if Bsqr is computed
    if (b2.size() != N)
    {   
        throw std::invalid_argument("bsq must be computed \n");
    }

    #pragma omp parallel
    {
        ARRAY beta_private;
        #pragma omp for nowait schedule(static) private(i, N)
        for (i=0; i<N; i++)
        {
            double _beta = PRIMS[iP][i] / b2[i];
            beta_private.push_back(_beta);
        }
        #pragma omp for schedule(static) ordered
        for(int i=0; i<omp_get_num_threads(); i++)
        {
            #pragma omp ordered
            beta.insert(beta.end(), beta_private.begin(), beta_private.end());
        }
    }
    return;
}

void HAMR_MHD::Getbfluid(ARRAY &bfluid0, ARRAY &bfluid1, ARRAY &bfluid2, ARRAY &bfluid3, ARRAY &b2,
                    const ARRAY2D &COORDS, const ARRAY2D &PRIMS)
{
    print("Computing bfluid");
    size_t i, N = COORDS[0].size();

    bfluid0.clear();
    bfluid1.clear();
    bfluid2.clear();
    bfluid3.clear();
    b2.clear();

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(12); // Use 4 threads for all consecutive parallel regions
    #pragma omp parallel
    {
        ARRAY bfluid0_private, bfluid1_private, bfluid2_private, bfluid3_private, b2_private;
        #pragma omp for nowait schedule(static) private(i,N)
        for(i=0; i<N; i++)
        {
            double x = COORDS[0][i], y = COORDS[1][i], z = COORDS[2][i];

            // Get KS coordinates
            double XKS[NDIM];
            CartToKS(x, y, z, XKS);

            // Now the metric matrices
            double gcov[NDIM][NDIM], gcon[NDIM][NDIM];
            GcovFunc(2, XKS, gcov);
            GconFunc(2, XKS, gcon);

            double B[NDIM] = {PRIMS[iB0][i], PRIMS[iB1][i], PRIMS[iB2][i], PRIMS[iB3][i]};
            double U[NDIM] = {PRIMS[iU0][i], PRIMS[iU1][i], PRIMS[iU2][i], PRIMS[iU3][i]};

            // Using BHAC equation 19 https://arxiv.org/pdf/1611.09720.pdf
            double alpha = 1. / sqrt(-gcon[0][0]);
            double lfac = alpha * U[0];
            double U_cov[NDIM];
            UpperToLower(2, XKS, U, U_cov);

            double _bf0 = 0., _bf1, _bf2, _bf3;
            #pragma omp simd reduction(+:_bf0)
            for (int ii=1; ii<NDIM; ii++)
            {
                _bf0 += B[ii] * U_cov[ii] / alpha;
            }
            _bf1 = (B[1] + alpha * _bf0 * U[1]) / lfac;
            _bf2 = (B[2] + alpha * _bf0 * U[2]) / lfac;
            _bf3 = (B[3] + alpha * _bf0 * U[3]) / lfac;

            // Taken from functions.c but commented out
            /*
            double _bf0 = 0.;
            #pragma omp simd reduction(+:_bf0)
            for (int ii=0; ii<NDIM; ii++)
            {
                _bf0 += (B[1] * U[ii] * gcov[1][ii] + B[2] * U[ii] * gcov[2][ii] + B[3] * U[ii] * gcov[3][ii]);
            }

            double _bf1 = (B[1] + _bf0 * U[1]) / (U[0]);
            double _bf2 = (B[2] + _bf0 * U[2]) / (U[0]);
            double _bf3 = (B[3] + _bf0 * U[3]) / (U[0]);
            */

            double _b_con[NDIM] = {_bf0, _bf1, _bf2, _bf3}, _b_cov[NDIM];
            UpperToLower(2, XKS, _b_con, _b_cov);
            double _b2 = _b_con[0]*_b_cov[0] + _b_con[1]*_b_cov[1] + _b_con[2]*_b_cov[2] + _b_con[3]*_b_cov[3];

            bfluid0_private.push_back(_bf0);
            bfluid1_private.push_back(_bf1);
            bfluid2_private.push_back(_bf2);
            bfluid3_private.push_back(_bf3);
            b2_private.push_back(_b2);
        }
        #pragma omp for schedule(static) ordered
        for(int i=0; i<omp_get_num_threads(); i++)
        {
            #pragma omp ordered
            {
                bfluid0.insert(bfluid0.end(), bfluid0_private.begin(), bfluid0_private.end());
                bfluid1.insert(bfluid1.end(), bfluid1_private.begin(), bfluid1_private.end());
                bfluid2.insert(bfluid2.end(), bfluid2_private.begin(), bfluid2_private.end());
                bfluid3.insert(bfluid3.end(), bfluid3_private.begin(), bfluid3_private.end());
                b2.insert(b2.end(), b2_private.begin(), b2_private.end());
            }
        }
    }
    return;
}
