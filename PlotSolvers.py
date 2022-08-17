import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 5, 5001)

# Suzuki-Trotter
# =============================================================================
# STmagDiff = np.loadtxt(r"ComparisonTests/ComponentDiffST_Diss.txt")
# STmagF = np.loadtxt(r"ComparisonTests/ComponentsSTForw_Diss.txt")
# STmagB = np.loadtxt(r"ComparisonTests/ComponentsSTBack_Diss.txt")
# STEnDiff = np.loadtxt(r"ComparisonTests/EnergyDiffST_Diss.txt")
# STEn = np.loadtxt(r"ComparisonTests/EnergyST_Diss.txt")
# =============================================================================

# =============================================================================
# ang = np.load(r"ComparisonTests/MaDudAnglesForw.npy")
# ang2 = np.load(r"ComparisonTests/MaDudAnglesBack.npy")
# phi = np.load(r"ComparisonTests/MaDudAzimuthForw.npy")
# phi2 = np.load(r"ComparisonTests/MaDudAzimuthBack.npy")
# STForwx = np.sin(ang) * np.cos(phi)
# STForwy = np.sin(ang) * np.sin(phi)
# STForwz = np.cos(ang)
# STBackx = np.sin(ang2) * np.cos(phi2)
# STBacky = np.sin(ang2) * np.sin(phi2)
# STBackz = np.cos(ang2)
# 
# STForw2 = np.column_stack((STForwx, STForwy, STForwz))
# STBack2 = np.column_stack((STBackx, STBacky, STBackz))
# =============================================================================

# =============================================================================
# STForw = np.loadtxt(r"ComparisonTests/ComponentsSTForw_OneSpin1.txt")
# STBack = np.loadtxt(r"ComparisonTests/ComponentsSTBack_OneSpin1.txt")
# 
# #dtt = np.loadtxt(r"ComparisonTests/ComponentsSTForwDiss_dt10.txt")
# dt = np.loadtxt(r"ComparisonTests/ComponentsSTForwDiss_dt1.txt")
# dt0 = np.loadtxt(r"ComparisonTests/ComponentsSTForwDiss_dt05.txt")
# dt1 = np.loadtxt(r"ComparisonTests/ComponentsSTForwDiss_dt01.txt")
# dt2 = np.loadtxt(r"ComparisonTests/ComponentsSTForwDiss_dt001.txt")
# dt3 = np.loadtxt(r"ComparisonTests/ComponentsSTForwDiss_dt0001.txt")
# =============================================================================
# =============================================================================
# dt4 = np.loadtxt(r"ComparisonTests/ComponentsSTForwDiss_dt00001.txt")
# dt5 = np.loadtxt(r"ComparisonTests/ComponentsSTForwDiss_dt000001.txt")
# =============================================================================


# Heun
# =============================================================================
# =============================================================================
# HeunmagDiff = np.loadtxt(r"ComparisonTests/ComponentDiffHeun_Diss.txt")
# HeunDiffCorr = np.loadtxt(r"ComparisonTests/ComponentDiffHeun_DissCorr.txt")
# =============================================================================
# HeunmagF = np.loadtxt(r"ComparisonTests/ComponentsHeunForw_Diss.txt")
# HeunmagB = np.loadtxt(r"ComparisonTests/ComponentsHeunBack_Diss.txt")
# HeunEnDiff = np.loadtxt(r"ComparisonTests/EnergyDiffHeun_Diss.txt")
# HeunEn = np.loadtxt(r"ComparisonTests/EnergyHeun_Diss.txt")
# =============================================================================

# =============================================================================
# HeunForw = np.loadtxt(r"ComparisonTests/ComponentsHeunForw_OneSpin1.txt")
# HeunBack = np.loadtxt(r"ComparisonTests/ComponentsHeunBack_OneSpin1.txt")
# 
# HeunForwCorr = np.loadtxt(r"ComparisonTests/ComponentsHeunForw_CorrectedOneSpin1.txt")
# HeunBackCorr = np.loadtxt(r"ComparisonTests/ComponentsHeunBack_CorrectedOneSpin1.txt")
# =============================================================================

# =============================================================================
# UppASDHeunForw = np.loadtxt(r"ComparisonTests/Heun3_2.out", skiprows = 7, 
#                            usecols = (4,5,6))
# =============================================================================

# SIB
# =============================================================================
# =============================================================================
# SIBmagDiff = np.loadtxt(r"ComparisonTests/ComponentDiffSIB_Diss.txt")
# SIBDiffCorr = np.loadtxt(r"ComparisonTests/ComponentDiffSIB_DissCorr.txt")
# =============================================================================
# SIBmagF = np.loadtxt(r"ComparisonTests/ComponentsSIBForw_Diss.txt")
# SIBmagB = np.loadtxt(r"ComparisonTests/ComponentsSIBBack_Diss.txt")
# SIBEnDiff = np.loadtxt(r"ComparisonTests/EnergyDiffSIB_Diss.txt")
# SIBEn = np.loadtxt(r"ComparisonTests/EnergySIB_Diss.txt")
# =============================================================================

# =============================================================================
# SIBForw = np.loadtxt(r"ComparisonTests/ComponentsSIBForw_OneSpin1.txt")
# SIBBack = np.loadtxt(r"ComparisonTests/ComponentsSIBBack_OneSpin1.txt")
# 
# SIBForwCorr = np.loadtxt(r"ComparisonTests/ComponentsSIBForw_CorrectedOneSpin1.txt")
# SIBBackCorr = np.loadtxt(r"ComparisonTests/ComponentsSIBBack_CorrectedOneSpin1.txt")
# =============================================================================

SIBdt1 = np.loadtxt(r"ComparisonTests/TimestepHeun_1.txt")
SIBdt2 = np.loadtxt(r"ComparisonTests/TimestepHeun_05.txt")
SIBdt3 = np.loadtxt(r"ComparisonTests/TimestepHeun_01.txt")
SIBdt4 = np.loadtxt(r"ComparisonTests/TimestepHeun_001.txt")
SIBdt5 = np.loadtxt(r"ComparisonTests/TimestepHeun_0001.txt")

# =============================================================================
# UppASDSIBForw = np.loadtxt(r"ComparisonTests/moment.ForwSIB3.out", skiprows = 7, 
#                            usecols = (4,5,6))
# =============================================================================

# Plots

# Analytical expression for single spin in 10 T external field 
# External field in #1: z direction, #2: -y direction. Spin always originally 
# along x direction
t2 = np.linspace(0, 60, 60001)
b = 0.1
a = 0.1760859644 * 10.0 / (1 + b**2)
sx = np.cos(a * t2) / np.cosh(b * a * t2)
sy = np.sin(a * t2) / np.cosh(b * a * t2)
sz = np.tanh(b * a * t2)

sx2 = np.cos(a * t2) / np.cosh(b * a * t2)
sy2 = -np.tanh(b * a * t2)
sz2 = np.sin(a * t2) / np.cosh(b * a * t2)

# =============================================================================
# t3 = np.linspace(0, 60, 120001)
# sx3 = np.cos(a * t3) / np.cosh(b * a * t3)
# sy3 = -np.tanh(b * a * t3)
# sz3 = np.sin(a * t3) / np.cosh(b * a * t3)
# 
# t4 = np.linspace(0, 60, 150001)
# sx4 = np.cos(a * t4) / np.cosh(b * a * t4)
# sy4 = -np.tanh(b * a * t4)
# sz4 = np.sin(a * t4) / np.cosh(b * a * t4)
# =============================================================================

# =============================================================================
# err = np.abs(dtt - np.column_stack((sx2[::10000], sy2[::10000], sz2[::10000])))
# logErr = np.log(np.amax(err, axis = 1))
# =============================================================================

# =============================================================================
# error = np.abs(dt - np.column_stack((sx2[::1000], sy2[::1000], sz2[::1000])))
# logError = np.log(np.amax(error, axis = 1))
# 
# error0 = np.abs(dt0 - np.column_stack((sx2[::500], sy2[::500], sz2[::500])))
# logError0 = np.log(np.amax(error0, axis = 1))
# 
# error1 = np.abs(dt1 - np.column_stack((sx2[::100], sy2[::100], sz2[::100])))
# logError1 = np.log(np.amax(error1, axis = 1))
# 
# error2 = np.abs(dt2 - np.column_stack((sx2[::10], sy2[::10], sz2[::10])))
# logError2 = np.log(np.amax(error2, axis = 1))
# 
# error3 = np.abs(dt3 - np.column_stack((sx2, sy2, sz2)))
# logError3 = np.log(np.amax(error3, axis = 1))
# =============================================================================

error = np.abs(SIBdt1 - np.column_stack((sx2[::1000], sy2[::1000], sz2[::1000])))
logError = np.log(np.amax(error, axis = 1))

error0 = np.abs(SIBdt2 - np.column_stack((sx2[::500], sy2[::500], sz2[::500])))
logError0 = np.log(np.amax(error0, axis = 1))

error1 = np.abs(SIBdt3 - np.column_stack((sx2[::100], sy2[::100], sz2[::100])))
logError1 = np.log(np.amax(error1, axis = 1))

error2 = np.abs(SIBdt4 - np.column_stack((sx2[::10], sy2[::10], sz2[::10])))
logError2 = np.log(np.amax(error2, axis = 1))

error3 = np.abs(SIBdt5 - np.column_stack((sx2, sy2, sz2)))
logError3 = np.log(np.amax(error3, axis = 1))

# =============================================================================
# error4 = np.abs(dt4[:60001:] - np.column_stack((sx2, sy2, sz2)))
# logError4 = np.log(np.amax(error4, axis = 1))
# 
# error5 = np.abs(dt5[:60001:] - np.column_stack((sx2, sy2, sz2)))
# logError5 = np.log(np.amax(error5, axis = 1))
# =============================================================================

plt.xlabel("Time [ps]")
plt.ylabel(r"$ln(max(|S_{an}^{\alpha} - S_{num}^{\alpha}|))$")
#plt.plot(t2[::10000], logErr, label = "dt = 10.0")
#plt.plot(t2[::1000], logError, label = "dt = 1.0, Heun")
plt.plot(t2[::500], logError0, label = "dt = 0.5, Heun")
plt.plot(t2[::100], logError1, label = "dt = 0.1, Heun")
plt.plot(t2[::10], logError2, label = "dt = 0.01, Heun")
plt.plot(t2, logError3, label = "dt = 0.001, Heun")
# =============================================================================
# plt.plot(t2, logError4, label = "dt = 0.0001")
# plt.plot(t2, logError5, label = "dt = 0.00001")
# =============================================================================
plt.legend()

# =============================================================================
# SIBerror = np.abs(SIBForw - np.column_stack((sx2, sy2, sz2)))
# SIBlogError = np.log(np.amax(SIBerror, axis = 1))
# 
# SIBerror2 = np.abs(SIBForwCorr - np.column_stack((sx4, sy4, sz4)))
# SIBlogError2 = np.log(np.amax(SIBerror2, axis = 1))
# 
# Heunerror = np.abs(HeunForw - np.column_stack((sx2, sy2, sz2)))
# HeunlogError = np.log(np.amax(Heunerror, axis = 1))
# 
# Heunerror2 = np.abs(HeunForwCorr - np.column_stack((sx3, sy3, sz3)))
# HeunlogError2 = np.log(np.amax(Heunerror2, axis = 1))
# 
# STerror = np.abs(STForw - np.column_stack((sx2, sy2, sz2)))
# STlogError = np.log(np.amax(STerror, axis = 1))
# =============================================================================



# =============================================================================
# STerror2 = np.abs(STForw2 - np.column_stack((sx, sy, sz)))
# STlogError2 = np.log(np.amax(STerror2, axis = 1))
# =============================================================================

#plt.plot(t2, np.log(np.abs(STForw[:, 1]+1)))

# =============================================================================
# plt.xlabel("Time [ps]")
# plt.ylabel(r"$ln(max(|S_{an}^{\alpha} - S_{num}^{\alpha}|))$")
# plt.plot(t2, STlogError, label = "ST")
# #plt.plot(t2, STlogError2, label = "ST_sph_only")
# plt.plot(t2, HeunlogError, label = "Heun")
# plt.plot(t2, SIBlogError, label = "SIB")
# plt.plot(t3, HeunlogError2, label = "Heun Adjusted")
# plt.plot(t4, SIBlogError2, label = "SIB Adjusted")
# plt.legend()
# =============================================================================


# =============================================================================
# plt.xlabel("Time [ps]")
# plt.ylabel("Spin Components Difference")
# plt.plot(t2, STBack[:, 0] - sx2, label = "Error x")
# #plt.plot(t2, HeunBack[:, 1] - sy2, label = "Error y")
# plt.plot(t2, STBack[:, 2] - sz2, label = "Error z")
# plt.legend()
# =============================================================================

# =============================================================================
# plt.xlabel("Time [ps]")
# plt.ylabel("Spin Components Difference")
# plt.plot(t2, SIBBack[:, 0] - sx, label = "Error x")
# plt.plot(t2, SIBBack[:, 1] - sy, label = "Error y")
# plt.plot(t2, SIBBack[:, 2] - sz, label = "Error z")
# plt.legend()
# =============================================================================

# =============================================================================
# plt.xlabel("Time [ps]")
# plt.ylabel("Energy Difference [meV]")
# #plt.plot(t, STEnDiff, label = "ST Energy")
# #plt.plot(t, HeunEnDiff, label = "Heun Energy")
# plt.plot(t, SIBEnDiff, label = "SIB Energy")
# plt.legend()
# =============================================================================

# =============================================================================
# plt.xlabel("Time [ps]")
# plt.ylabel(r"$ln(max(|S_{forw}^{\alpha} - S_{back}^{\alpha}|))$")
# 
# STspin = np.log(np.amax(np.abs(STmagDiff[:, 0:5]), axis = 1))
# 
# Heunspin = np.log(np.amax(np.abs(HeunmagDiff[:, 0:5]), axis = 1))
# HeunSpinCorr = np.log(np.amax(np.abs(HeunDiffCorr[:, 0:5]), axis = 1))
# 
# SIBspin = np.log(np.amax(np.abs(SIBmagDiff[:, 0:5]), axis = 1))
# SIBspinCorr = np.log(np.amax(np.abs(SIBDiffCorr[:, 0:5]), axis = 1))
# 
# t3 = np.linspace(0, 5, 40001)
# 
# plt.plot(t, STspin, label = "ST")
# plt.plot(t, Heunspin, label = "Heun")
# plt.plot(t, SIBspin, label = "SIB")
# plt.plot(t3, HeunSpinCorr, label = "Heun Adjusted")
# plt.plot(t3, SIBspinCorr, label = "SIB Adjusted")
# plt.legend()
# =============================================================================
