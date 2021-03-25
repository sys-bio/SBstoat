MODEL =  \
'''
function Fiii(ri1, ri2, ri3, kf, kr, i1, i2, i3, s, p, Kmi1, Kmi2, Kmi3, Kms, Kmp, wi1, wi2, wi3, ms, mp)
    ((ri1+(1-ri1)*(1/(1+i1/Kmi1)))^wi1) * ((ri2+(1-ri2)*(1/(1+i2/Kmi2)))^wi2) * ((ri3+(1-ri3)*(1/(1+i3/Kmi3)))^wi3) * (kf*(s/Kms)^ms-kr*(p/Kmp)^mp)/((1+(s/Kms))^ms+(1+(p/Kmp))^mp-1)
end
function Fi(ri, kf, kr, i, s, p, Kmi, Kms, Kmp, wi, ms, mp)
    ((ri+(1-ri)*(1/(1+i/Kmi)))^wi)*(kf*(s/Kms)^ms-kr*(p/Kmp)^mp)/((1+(s/Kms))^ms+(1+(p/Kmp))^mp-1)
end
function Fa(ra, kf, kr, a, s, p, Kma, Kms, Kmp, wa, ms, mp)
    ((ra+(1-ra)*((a/Kma)/(1+a/Kma)))^wa)*(kf*(s/Kms)^ms-kr*(p/Kmp)^mp)/((1+(s/Kms))^ms+(1+(p/Kmp))^mp-1)
end
function Hi(V, K, n, K2, n2, t, s)
    (1/((s^n2) + (K2^n2))) * ((V * n * (K^n) * (t^(n-1))) / (((K^n) + (t^n))^2))
end
function F0(kf, kr, s, p, Kms, Kmp, ms, mp)
    (kf*(s/Kms)^ms-kr*(p/Kmp)^mp)/((1+(s/Kms))^ms+(1+(p/Kmp))^mp-1)
end
function Ha(V, K, n, K2, n2, t, s)
    ((s^n2)/((s^n2) + (K2^n2))) * ((V * n * (K^n) * (t^(n-1))) / (((K^n) + (t^n))^2))
end

model coarseEGF_0_0()


// Reactions
FreeLigand: -> L; Fa(ra_0, kf_0, kr_0, Lp, E, L, Kma_0, Kms_0, Kmp_0, wa_0, ms_0, mp_0);
Phosphotyrosine: -> P; Fi(ri_0, kf_1, kr_1, Mig6, L, P, Kmi_1, Kms_1, Kmp_1, wi_1, ms_1, mp_1);
Ras: -> R; Fiii(ri1_1, ri2_1, ri3_1, kf_2, kr_2, Spry2, P, E, P, R, Kmi1_2, Kmi2_2, Kmi3_2, Kms_2, Kmp_2, wi1_2, wi2_2, wi3_2, ms_2, mp_2);
Erk: -> E; F0(kf_3, kr_3, R, E, Kms_3, Kmp_3, ms_3, mp_3);
Precursor: -> Lp; kLp*C1;
Spry: -> Spry2; Ha(Va, Ka, na, Ka2, na2, t, L);
Mig: -> Mig6; Hi(Vi, Ki, ni, Ki2, ni2, t, L);

// Species IVs
Lp = 0;
E = 0;
L = 0.165;
Mig6 = 0;
P = 0;
Spry2 = 0;
R = 0;
C1 = 1;

// Parameter values
ra_0 = 1;
kf_0 = 1;
kr_0 = 1;
Kma_0 = 1;
Kms_0 = 1;
Kmp_0 = 1;
wa_0 = 1;
ms_0 = 1;
mp_0 = 1;
ri_0 = 1;
kf_1 = 1;
kr_1 = 1;
Kmi_1 = 1;
Kms_1 = 1;
Kmp_1 = 1;
wi_1 = 1;
ms_1 = 1;
mp_1 = 1;
ri1_1 = 1;
ri2_1 = 1;
ri3_1 = 1;
kf_2 = 1;
kr_2 = 1;
Kmi1_2 = 1;
Kmi2_2 = 1;
Kmi3_2 = 1;
Kms_2 = 1;
Kmp_2 = 1;
wi1_2 = 1;
wi2_2 = 1;
wi3_2 = 1;
ms_2 = 1;
mp_2 = 1;
kf_3 = 1;
kr_3 = 1;
Kms_3 = 1;
Kmp_3 = 1;
ms_3 = 1;
mp_3 = 1;
kLp = 0.00252;
Va = 23.0314252;
Ka = 10153.3423;
na = 2.52290896;
Ka2 = 20.134033969;
na2 = 8.48874275;
t := time;
Vi = 0.790702084;
Ki = 8461.68691;
ni = 1.49168606;
Ki2 = 0.160718999;
ni2 = 0.975509826;

end
'''

PARAMETERS = ["ra_0", "kf_0", "kr_0", "Kma_0", "Kms_0", "Kmp_0", "wa_0", "ms_0",
                                           "mp_0", "ri_0", "kf_1", "kr_1", "Kmi_1", "Kms_1", "Kmp_1", "wi_1",
                                           "ms_1", "mp_1", "ri1_1", "ri2_1", "ri3_1", "kf_2", "kr_2",
                                           "Kmi1_2", "Kmi2_2", "Kmi3_2", "Kms_2", "Kmp_2", "wi1_2", "wi2_2", "wi3_2",
                                           "ms_2", "mp_2", "kf_3", "kr_3", "Kms_3", "Kmp_3", "ms_3", "mp_3"]
