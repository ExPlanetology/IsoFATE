# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plotting script for isofate_coupler."""

import matplotlib.pyplot as plt
import numpy as np

from isofate.constants import const
from isofate.initial_condition import ProtosolarInitialCondition
from isofate.utils import gravitational_acceleration, scale_height


def isoplot(output, parameters):
    """The diagnostic figure of isofate/sim.py: a 3x3 grid of escape fluxes, the He fraction,
    atmospheric moles, planet radius, surface temperature (annotated with final values),
    envelope mass, gravitational potential, mass loss and atmospheric mass fraction.

    Moved from the end of sim.py. It replaces the earlier ``isoplot(sol, n_atmodeller, ...)``
    (removed), which drew the same figure from a raw solution dict with only H, He, D, O and C.

    Args:
        output: The ``isofate.output.Output`` of ``isocalc``
        parameters: The ``isofate.parameters.Parameters`` the run used

    Returns:
        The figure. Show it with ``plt.show()``.
    """
    # TODO refactor notes, collected from this function and the earlier ``isoplot(sol, ...)``:
    # - Panel 2 ("D/H, O/H" header): it was meant to show the fractionation ratio, with D/H or
    #   O/H [Solar] as commented alternatives to x_He. Make it selectable. Note the commented code
    #   never divides by the solar value, despite its "[Solar]" label.
    # - Analytic overlays (panels 2 and 3, commented): x2sub, x2sub2, x2super, analytic_soln, N1
    #   and N1_mod for the "fix phi subcritical/supercritical" mechanisms. These validated against
    #   analytic solutions, but none of these functions exist in the package any more.
    # - Analytic radii (earlier ``isoplot(sol, ...)`` only, never plotted): it computed
    #   Rp = min(R_rocky + R_env + R_atm, R_Bondi, R_Hill) with a solar mu, plus A = 4 pi Rp^2 and
    #   M_atm = Mp f_atm, as inputs to those overlays. "Rp = r_core if rad_evol is False for the
    #   analytics to match isofate". sim.py still computes this, with bondi_radius on the final
    #   composition.
    # - Panel 9 "model parameters" (commented): a text panel of f_atm, composition, Mp, Rp(t=0),
    #   F_XUV(t=0), P, Fp, Teq and the mechanism. All are in ``parameters`` now. It could replace
    #   the crowded panel 5 annotations, which sit at hard-coded data positions (x = 2e6 yr) and
    #   drop off-screen when the first output time is later.
    # - Envelope radius panel (commented, ``renv``): no longer output, but ``Ratm`` (convective
    #   envelope thickness) is, so the panel could come back.
    # - Compact layout: a 1x3 grid of just phi, x2 and N1/N2 (commented ``plt.subplots(1, 3)``).
    # - Dead remnants to remove: ``adaptt_a``/'time_adapted', the commented prints (sim.py prints
    #   them), the duplicate ``cross`` axvline on ax1 after panel 2, and the unused ``N_tot``,
    #   ``Ts_analytic`` and ``x1_a`` (once the critical lines go).
    sol = output.to_dict(to_numpy=True)
    Mp = parameters.system.planet.mass
    T = parameters.system.equilibrium_temperature
    # the run's own species masses [kg/atom] and binary diffusion coefficients, as the model uses
    atomic_mass = parameters.isofate_species.mass_by_symbol
    binary_diffusion = parameters.binary_diffusion_coefficients
    protosolar = ProtosolarInitialCondition.PROTOSOLAR_MOLE_RATIOS_TO_H
    # initial atmospheric abundances [atoms]
    N_H, N_He, N_D, N_O, N_C, N_N, N_S = (float(n) for n in parameters.initial_abundances())

    t_a = sol["time"]
    # adaptt_a = sol['time_adapted']
    rp_a = sol["Rp"]
    # renv_a = sol['renv']
    menv_a = sol["Matm"]
    vpot_a = sol["Vpot"]
    fenv_a = sol["fatm"]
    mloss_a = sol["Mloss"]
    phi_a = sol["phi"]
    phic_a = sol["phic"]
    NH_a = sol["N_H"]
    NHe_a = sol["N_He"]
    ND_a = sol["N_D"]
    NO_a = sol["N_O"]
    NC_a = sol["N_C"]
    NN_a = sol["N_N"]
    NS_a = sol["N_S"]
    x1_a = sol["x1"]
    x2_a = sol["x2"]
    PhiH_a = sol["Phi_H"]
    PhiHe_a = sol["Phi_He"]
    PhiD_a = sol["Phi_D"]
    PhiO_a = sol["Phi_O"]
    PhiC_a = sol["Phi_C"]
    PhiN_a = sol["Phi_N"]
    PhiS_a = sol["Phi_S"]
    Ts_analytic = sol["T_surf_analytic"]
    Ts_atmod = sol["T_surf_atmod"]
    final_composition: dict[str, float] | None = None  # final mole fractions, with Atmodeller
    if output.atmodeller is not None:
        # gas-phase moles of each molecule at the last equilibration (the last Atmodeller batch row)
        atmodeller_final = output.atmodeller.to_dict(
            output_format="elements_species", to_numpy=True
        )
        n_H2O, n_H2, n_O2, n_CO2, n_CO, n_CH4, n_N2, n_S2, n_H2O4S, n_SO2 = (
            atmodeller_final[name]["gas"]["number_moles"].ravel()[-1]
            for name in (
                "H2O_g",
                "H2_g",
                "O2_g",
                "CO2_g",
                "CO_g",
                "CH4_g",
                "N2_g",
                "S2_g",
                "H2O4S_g",
                "O2S_g",
            )
        )
        n_total_atm = (
            n_H2O
            + n_H2
            + n_O2
            + n_CO2
            + n_CO
            + n_CH4
            + n_N2
            + n_S2
            + n_H2O4S
            + n_SO2
            + NHe_a[-1] / const.avogadro
        )
        x_CO2 = n_CO2 / n_total_atm
        x_CO = n_CO / n_total_atm
        x_CH4 = n_CH4 / n_total_atm
        x_H2 = n_H2 / n_total_atm
        x_O2 = n_O2 / n_total_atm
        x_H2O = n_H2O / n_total_atm
        x_N2 = n_N2 / n_total_atm
        x_S2 = n_S2 / n_total_atm
        x_H2O4S = n_H2O4S / n_total_atm
        x_SO2 = n_SO2 / n_total_atm
        N_tot = N_H + N_He + N_D + N_O + N_C + N_N + N_S
        N_tot_molecular = (
            n_H2O + n_H2 + n_O2 + n_CO + n_CO2 + n_N2 + n_S2 + n_H2O4S + n_SO2 + n_CH4
        ) * const.avogadro + NHe_a[-1]

        #### This is all wrong, you need to fix with the actual output. You're just using initial conditions here.
        # x_H = N_H/N_tot
        # x_He = N_He/N_tot
        x_He_molecular = NHe_a[-1] / N_tot_molecular
        # x_D = N_D/N_tot
        # x_O = N_O/N_tot
        # x_C = N_C/N_tot
        # x_N = N_N/N_tot
        # x_S = N_S/N_tot
        # print('x_H2O =', x_H2O)
        # print('x_H2 =', x_H2)
        # print('x_He =', x_He_molecular)
        # print('x_O2 =', x_O2)
        # print('x_CO2 =', x_CO2)
        # print('x_CO =', x_CO)
        # print('x_CH4 =', x_CH4)
        # print('x_N2 =', x_N2)
        # print('x_S2 =', x_S2)
        # print('x_H2O4S =', x_H2O4S)
        # print('x_SO2 =', x_SO2)
        # final mole fractions for the annotations below (He first)
        final_composition = {
            "He": x_He_molecular,
            "O2": x_O2,
            "H2O": x_H2O,
            "CO2": x_CO2,
            "CO": x_CO,
            "CH4": x_CH4,
            "H2": x_H2,
            "N2": x_N2,
            "S2": x_S2,
            "H2O4S": x_H2O4S,
            "SO2": x_SO2,
        }
    # calculate mass fraction of He

    # He mass fraction of the atmosphere, over every tracked species
    atmosphere_mass = sum(sol[f"N_{symbol}"] * mass for symbol, mass in atomic_mass.items())
    Y = NHe_a * atomic_mass["He"] / atmosphere_mass

    plt.rcParams["figure.figsize"] = (14, 8)
    plt.rcParams.update({"font.size": 14})
    plt.rcParams.update({"axes.titlesize": 11})

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex=False)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex = False) # use for just phi, x2, and N1/N2 plots
    plt.subplots_adjust(wspace=0.3)

    # phi
    # gravity at the escape radius (output Rp) and H scale height, as the model evaluates them
    g = gravitational_acceleration(Mp, rp_a)
    H_H = scale_height(T, atomic_mass["H"], g)  # H scale height [m]
    ax1.plot(t_a * const.s2yr, PhiH_a * atomic_mass["H"], color="black", label="H flux")
    ax1.plot(t_a * const.s2yr, PhiHe_a * atomic_mass["He"], color="grey", label="He flux")
    ax1.plot(t_a * const.s2yr, phic_a, "--", color="grey", label="He critical")
    if ND_a[0] != 0:
        ax1.plot(t_a * const.s2yr, PhiD_a * atomic_mass["D"], color="orangered", label="D flux")
        ax1.plot(
            t_a * const.s2yr,
            binary_diffusion.get("H", "D", T) * x1_a * (atomic_mass["D"] - atomic_mass["H"]) / H_H,
            "--",
            color="orangered",
            label="D critical",
        )  # D/H critical flux
    if NO_a[0] != 0:
        ax1.plot(t_a * const.s2yr, PhiO_a * atomic_mass["O"], color="green", label="O flux")
        ax1.plot(
            t_a * const.s2yr,
            binary_diffusion.get("H", "O", T) * x1_a * (atomic_mass["O"] - atomic_mass["H"]) / H_H,
            "--",
            color="green",
            label="O critical",
        )  # O/H critical flux
    if NC_a[0] != 0:
        ax1.plot(t_a * const.s2yr, PhiC_a * atomic_mass["C"], color="gold", label="C flux")
        ax1.plot(
            t_a * const.s2yr,
            binary_diffusion.get("H", "C", T) * x1_a * (atomic_mass["C"] - atomic_mass["H"]) / H_H,
            "--",
            color="gold",
            label="C critical",
        )  # C/H critical flux
        # if NN_a[0] != 0:
        ax1.plot(t_a * const.s2yr, PhiN_a * atomic_mass["N"], color="blue", label="N flux")
        ax1.plot(
            t_a * const.s2yr,
            binary_diffusion.get("H", "N", T) * x1_a * (atomic_mass["N"] - atomic_mass["H"]) / H_H,
            "--",
            color="blue",
            label="N critical",
        )  # N/H critical flux
    if NS_a[0] != 0:
        ax1.plot(t_a * const.s2yr, PhiS_a * atomic_mass["S"], color="purple", label="S flux")
        ax1.plot(
            t_a * const.s2yr,
            binary_diffusion.get("H", "S", T) * x1_a * (atomic_mass["S"] - atomic_mass["H"]) / H_H,
            "--",
            color="purple",
            label="S critical",
        )  # S/H critical flux
    ax1.plot(t_a * const.s2yr, phi_a, ":", color="mediumslateblue", label="total flux")
    cross = np.where(np.abs(phi_a - phic_a) < 1e-12)
    if len(cross[0]) != 0:
        ax1.axvline(t_a[cross][0] * const.s2yr, ls="--", lw=0.5, color="coral")
        print("critical flux at ", t_a[cross][0] * const.s2yr / 1e9, "Gyr")
    ax1.set_ylabel("phi [kg m$^{-2}$ s$^{-1}$]", labelpad=2)
    ax1.legend(fontsize=8.5, frameon=False, framealpha=0.7, loc=3, ncol=3)
    ax1.set_ylim(PhiH_a[0] * atomic_mass["H"] / 1e8, PhiH_a[0] * atomic_mass["H"] * 100)
    ax1.set_yscale("log")

    # D/H, O/H
    ax2.loglog(t_a * const.s2yr, x2_a, color="maroon", label="IsoFATE")
    ax2.set_ylabel("x$_{He}$ (atom fraction)", labelpad=2)  # x2: He over every tracked species
    # if ND_a[0] != 0:
    #     ax2.loglog(t_a*s2yr, ND_a/NH_a, color = 'maroon', label = 'IsoFATE')
    #     ax2.set_ylabel('D/H [Solar]', labelpad = 2)
    # if NO_a[0] != 0:
    #     ax2.loglog(t_a*s2yr, NO_a/NH_a, color = 'maroon', label = 'IsoFATE')
    #     ax2.set_ylabel('O/H [Solar]', labelpad = 2)

    # ax2.set_title('N$_2$/N$_1$ [xSolar]')
    # ax2.set_xlabel('Time [yr]')
    # if mechanism == 'fix phi subcritical':
    #     x2_analytic = x2sub(x2_0, tau, t_a*s2yr)
    #     # x2_analytic = x2sub2(x2_0, N1_0, A, Phi1, t_a*s2yr)
    #     ax2.loglog(t_a*s2yr, x2_analytic/dh_a[0,0], '--', color = 'deeppink', zorder = 10, label = 'analytic')
    #     ax2.legend(shadow = True)
    # elif mechanism == 'fix phi supercritical':
    #     x2_analytic = x2super(x2_0, Mp, Rp, T, Phi, tau/s2yr, t_a)
    #     ax2.loglog(t_a*s2yr, x2_analytic/dh_a[0,0], '--', color = 'deeppink', zorder = 10, label = 'analytic')
    #     ax2.legend(shadow = True)
    # else:
    # x2_analytic = analytic_soln(x2_0, N1_0, mu1, mu2, x1_a[0], x2_a[0], Mp, rp_a[0]*Re, fenv_a[0], T, phi_a[0], phic_a[0], t_a)
    # ax2.loglog(t_a*s2yr, x2_analytic[0]/dh_a[0,0], '--', color = 'deeppink', zorder = 10, label = 'analytic')
    # ax2.legend(shadow = True)
    # ax2.set_title('atmospheric fractionation')
    if len(cross[0]) != 0:
        ax1.axvline(t_a[cross][0] * const.s2yr, ls="--", lw=0.5, color="coral")
    # ax2.set_xlabel('time [yr]')

    # N_x
    ax3.plot(t_a * const.s2yr, NH_a / const.avogadro, color="black", label="N$_{H}$")
    # ax3.set_xlabel('Time [yr]')
    if len(cross[0]) != 0:
        ax3.axvline(t_a[cross][0] * const.s2yr, ls="--", lw=0.5, color="coral")
    if NHe_a[0] != 0:
        ax3.plot(t_a * const.s2yr, NHe_a / const.avogadro, color="grey", label="N$_{He}$")
    if ND_a[0] != 0:
        ax3.plot(t_a * const.s2yr, ND_a / const.avogadro, color="orangered", label="N$_D$")
    if NO_a[0] != 0:
        ax3.plot(t_a * const.s2yr, NO_a / const.avogadro, color="green", label="N$_{O}$")
    if NC_a[0] != 0:
        ax3.plot(t_a * const.s2yr, NC_a / const.avogadro, color="gold", label="N$_{C}$")
    if NN_a[0] != 0:
        ax3.plot(t_a * const.s2yr, NN_a / const.avogadro, color="blue", label="N$_{N}$")
    if NS_a[0] != 0:
        ax3.plot(t_a * const.s2yr, NS_a / const.avogadro, color="purple", label="N$_{S}$")
    # if mechanism == 'fix phi subcritical' or mechanism == 'fix phi supercritical':
    #     ax3.loglog(t_a*s2yr, N1(t_a, A, Phi, N1_0)/avogadro, '--', color = 'skyblue', zorder = 10, label = 'analytic N$_{H}$') # changed H_0 to N1_0
    #     ax3.legend(shadow = True)
    # else:
    # ax3.loglog(t_a*s2yr, N1(t_a, A, phi_c/mu1, N1_0), '--', color = 'deeppink', zorder = 10, label = 'analytic') # changed H_0 to N1_0
    # ax3.loglog(t_a*s2yr, N1_mod(t_a, 4*np.pi*r_core**2, phi_c/mu1, N1_0), '--', color = 'deeppink', zorder = 10, label = 'analytic') # changed H_0 to N1_0
    # ax3.set_title('hydrogen number')
    ax3.set_ylabel("atmospheric moles", labelpad=2)
    ax3.legend(frameon=False, fontsize=9, loc=4)
    ax3.set_ylim(NH_a[0] / const.avogadro / 1e12, NH_a[0] / const.avogadro * 100)
    ax3.set_yscale("log")
    # ax3.set_xlabel('time [yr]')

    # planet radius
    ax4.loglog(t_a * const.s2yr, rp_a / const.Re, color="mediumseagreen")
    # ax4.set_title('planetary radius')
    ax4.set_ylabel(r"radius [R$_\oplus$]", labelpad=2)
    # ax4.set_xlabel('time [yr]')

    # envelope radius
    # ax5.loglog(t_a*s2yr, renv_a[0]*1000, color = 'skyblue')
    # # ax5.set_title('envelope radius')
    # ax5.set_ylabel('R$_{env}$ [km]', labelpad = 2)
    # #ax5.set_xlabel('time [yr]')

    ax5.plot(t_a * const.s2yr, Ts_atmod, color="mediumslateblue")
    ax5.set_xscale("log")
    ax5.set_ylabel("surface temp [K]", labelpad=2)
    ax5.set_ylim(-100, 6100)
    ax5.annotate(
        f"Rp = {round(rp_a[-1] / const.Re, 2)} Re, Mp = {round(Mp / const.Me, 2)} Me, Teq = {round(T, 0)} K",
        (2e6, 5500),
        fontsize=8,
    )
    ax5.annotate("final fatm:" + str(round(fenv_a[-1], 6)), (2e6, 5100), fontsize=8)
    ax5.annotate(
        "final D/H:" + str(round(ND_a[-1] / NH_a[-1] / protosolar["D"], 2)) + " [Solar]",
        (2e6, 4700),
        fontsize=8,
    )
    if final_composition is not None:
        ax5.annotate(
            "final He concn:"
            + str(round(final_composition["He"], 4))
            + " mol/mol, "
            + str(round(Y[-1], 4))
            + " kg/kg",
            (2e6, 4300),
            fontsize=8,
        )
        molecules = [name for name in final_composition if name != "He"]
        for i, name in enumerate(molecules, start=1):
            ax5.annotate(
                f"final {name} concn:" + str(round(final_composition[name], 4)),
                (2e6, 4300 - 400 * i),
                fontsize=8,
            )
    else:
        ax5.annotate(
            "final X_He (atom fraction) =" + str(x2_a[-1]),  # over every tracked species
            (2e6, 4300),
            fontsize=8,
        )
    # envelope mass
    ax6.loglog(t_a * const.s2yr, menv_a / const.Me, color="midnightblue")
    # ax6.set_title('envelope mass')
    ax6.set_ylabel(r"M$_{env}$ [M$_\oplus$]", labelpad=2)
    # ax6.set_xlabel('time [yr]')

    # grav potential
    ax7.loglog(t_a * const.s2yr, vpot_a, color="orange")
    # ax7.set_title('gravitational potential')
    ax7.set_ylabel("V$_{pot}$ [J/kg]", labelpad=2)
    ax7.set_xlabel("time [yr]")

    # mass loss
    ax8.loglog(t_a * const.s2yr, mloss_a, color="crimson")
    # ax8.set_title('mass loss per time step')
    ax8.set_ylabel(r"$\Delta$ mass [kg]", labelpad=2)
    ax8.set_xlabel("time [yr]")

    # f_env or system parameters
    ax9.set_xlabel("time [yr]")

    ### model parameters

    # if adaptive_steps == False:
    # ax9.annotate('f_atm ='+str(f_atm)+',', (1e8, 0.9))
    # ax9.annotate('comp:'+str(species), (3e9, 0.9))
    # ax9.annotate('Mp ='+str(round(Mp/Me, 2))+' M$_\oplus$', (1e8, 0.78))
    # ax9.annotate('Rp(t=0) ='+str(round(Rp/Re, 2))+' R$_\oplus$', (1e8, 0.66))
    # ax9.annotate('F$_{XUV}$(t=0) ='+str(round(F0, 2))+' W/m2', (1e8, 0.54))
    # ax9.annotate('P ='+str(round(P*s2day, 2))+' days', (1e8, 0.42))
    # ax9.annotate('Fp ='+str(round(Fp, 2))+' W/m2', (1e8, 0.30))
    # ax9.annotate('Teq ='+str(round(T, 2))+' K', (1e8, 0.18))
    # ax9.annotate('mechanism ='+str(mechanism), (1e8, 0.06))
    # ax9.tick_params(left = 'False')
    # ax9.set_yticks([])

    ### atmospheric mass fraction

    ax9.set_ylabel("f_env [%]", labelpad=2)
    ax9.plot(t_a * const.s2yr, fenv_a * 100, color="gold")
    ax9.set_xscale("log")

    plt.tight_layout()

    return fig
