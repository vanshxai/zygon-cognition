"""Works out the unit (dimension) of every Frost Math variable from the formulas themselves.   python units_infer.py  ->  model/math_solo/units.json
A few unambiguous variables are given (SEEDS). Every formula is then a set of equations: both sides must have the same dimension, added terms must match,
and the inside of exp / log / sin must have none. Solved exactly (fractions) by elimination. A variable the equations cannot pin down is left 'unknown', never guessed.
A formula that contradicts the others is listed as 'does not balance' and is left out of the solution, so one bad formula cannot spoil the rest."""
import json, os, re, sys
from fractions import Fraction as Fr
import sympy as sp
sys.stdout.reconfigure(encoding="utf-8")
import units as U
import frostmath as fm

D = lambda s: U.d(**{m[0] if m[0] != "T" or True else m[0]: Fr(m[1] or 1) for m in []})


def dm(spec):
    """'M L2 T-3 I-1' -> dimension tuple"""
    out = dict.fromkeys(U.DIMS, Fr(0))
    for tok in spec.split():
        m = re.match(r"^(TH|[MLTINJ])(-?\d+(?:/\d+)?)?$", tok); out[m.group(1)] += Fr(m.group(2) or 1)
    return tuple(out[k] for k in U.DIMS)


SEEDS = {
    "VOLT": ("M L2 T-3 I-1", "V V1 V2 E E1 E2 V_L V_in V_out V_dd V_ph3 V_sync V_nom_batt Vin_buck Vout_buck Vout_boost Vz_zener Vd_diode V_rms_sine V_peak_sine Vin_instr Vout_instr V_pp_sine"),
    "CURRENT": ("I", "I I1 I2 I_L I_total I_cpu I_wifi I_batt I_load I_cable I_rate_batt"),
    "OHM": ("M L2 T-3 I-2", "R R0 R1 R2 R_load R_int_batt R_th_dummy R_cable_per_km_dummy R_555a R_555b R_filt R_pull R_brake_dummy"),
    "ACCEL": ("L T-2", "g_grav"), "SPEED": ("L T-1", "c_light v_hyd v_belt v_cyl v_torricelli"),
    "MU0": ("M L T-2 I-2", "mu0"), "SIGMA": ("M T-3 TH-4", "sigma_sb"),
    "DENSITY": ("M L-3", "rho rho_air rho_hyd rho_fluid_b rho_manometer rho_air_wind"),
    "TEMP": ("TH", "T_amb T_kelvin T_chip T_hot_rad T_cold_rad T_batt_amb T_cell_pv T_ref_pv T_final T_temp T0_temp T0_therm T_clk_dummy"),
    "TEMPDIFF": ("TH", "dT_hvac dT_hx dT1_hx dT2_hx"),
    "TIME": ("T", "t_time t_batt t_setup t_solidify t_clk_to_q t_pd_gate ts_settling tau_thermal t_inst t_relay"),
    "FREQ": ("T-1", "omega omega_s omega_n f f_ac f_rf f_crystal f_cpu f_clk_digital f_sw_cmos f_ripple f_step_rate"),
    "AREA2ND": ("L4", "I_area J_polar"), "LENGTH2": ("L", "D2_impeller Dp_gear Dg_gear D_workpiece"), "SPEED2": ("L T-1", "u2_impeller v_pitch_gear v_comm Vu2_whirl"), "FREQ2": ("T-1", "N_pump N_pump2 N_spindle N_s N_crit_shaft N_rpm_stepper RPM"), "SPEED3": ("L T-1", "Vc_cutting"), "LENGTH": ("L", "L_im D_im d_conc b_conc d_cut d_rivet L_weld t_weld throat_weld L d D_pipe L_pipe L_beam Lspan_beam L_rotor D_rotor D_comm D_coil d_wire D_vessel t_vessel L_ant dist_rf h_hyd h_manometer h_suction h_torricelli h_atm"),
    "AREA": ("L2", "A A_cable A_orifice A_cyl A_hyd A_rotor A_cond_area A_brush"),
    "MASS": ("M", "m m_sys m_gear m_batt"),
    "PRESSURE": ("M L-1 T-2", "P_hyd P_total_hvac Pv_vapor dP_valve E_mod G_spring P_internal P_manometer sigma_bend sigma_hoop sigma_vm tau sigma_a sigma_m sigma_eng sigma_true sigma_x sigma_y sigma_1_transform sigma_2_transform sigma_principal sigma_long sigma_lewis sigma_agma sigma_allow_col Sut_ultimate Se_endurance Se_corrected fy_conc fc_conc tau_xy tau_key tau_rivet tau_weld tau_spring tau_m_shaft"),
    "TORQUE": ("M L2 T-2", "M_bend M_max_udl T_brake T_clutch T_pump T_thrust_dummy"), "LINELOAD": ("M T-2", "w_udl"),
    "FORCE": ("M L T-2", "F F_beam F_hyd F_buoy F_cyl"),
    "ENERGY": ("M L2 T-2", "KE E_batt_wh_dummy"),
    "POWER": ("M L2 T-3", "P_out P_in Q_cooling_load P_lamp"),
    "VOLFLOW": ("L3 T-1", "Q_hyd Q_pump Q_orifice"), "VOL": ("L3", "Vol Vol_casting Vol_disp"),
    "MASSFLOW": ("M T-1", "m_dot_air_hvac m_dot_air m_dot_hx m_dot_fuel"),
    "HEATCAP": ("L2 T-2 TH-1", "cp_air_hvac Cp cp_hx"),
    "HTC": ("M T-3 TH-1", "h_conv"), "KTH": ("M L T-3 TH-1", "k_thermal_cond k_cond"),
    "VISC": ("M L-1 T-1", "mu_visc"), "PERM": ("M L T-2 I-2", "mu0"),
    "NONE": ("", "theta_step VR_sync_pct err_pct Cd_drag Cl_lift Cl_max Cd_orifice Cp_betz Cp_betz_limit CF_crest Kt_stress Ks_service Kv_dynamic K_minor L_D_ratio Gain_db SNR_db RL_db Gm_margin Av_opamp Av_instr beta_bjt epsilon_strain epsilon_eng epsilon_true eps_hx eps_rad F_ar Pf pf0_im eta_luminous eta_pv eta_overall_hyd eta_mech_hyd eta_vol eta_brayton eta_rankine_steam eta_otto eta_diesel eta eta_im eta_carnot eta_t pf_im pf_1ph pf_3ph s_slip gamma_gas nu_poisson mu_belt mu_brake mu_clutch Re_hyd Nu_nusselt Bi_biot f_darcy COP_carnot COP_heatpump COP_refrig load_frac_im SoC_batt DoD_batt W_humidity_ratio gamma_sg n_safety n_poles n_poles_im N_steps_rev N_segments N_fanout n_moles_gas_dummy zeta_damp zeta_2nd Mp_overshoot theta_wrap theta_cam theta_transform m_phases_im k_w_im Kf_shaft Gamma_refl G_ol"),
}
SEED = {}
for _, (spec, names) in SEEDS.items():
    for n in names.split(): SEED[n] = dm(spec)

FUN = (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh)


class Skip(Exception): pass


def add(a, b, k=1):
    r = dict(a)
    for v, c in b.items():
        r[v] = r.get(v, 0) + k * c
        if r[v] == 0: del r[v]
    return r


def lin(e, cons, flags):
    """dimension of e as a combination of variable dimensions; cons collects things that must be dimensionless / equal"""
    if e.is_Symbol: return {str(e): Fr(1)}
    if e.is_Number or e in (sp.pi, sp.E): return {}
    if e.is_Mul:
        r = {}
        for a in e.args: r = add(r, lin(a, cons, flags))
        return r
    if e.is_Pow:
        b, x = e.args
        if x.is_Number: return {v: c * Fr(int(x.p), int(x.q)) for v, c in lin(b, cons, flags).items()}
        cons.append(lin(b, cons, flags)); cons.append(lin(x, cons, flags)); return {}
    if e.is_Add:
        first, one = None, False
        for a in e.args:
            if a.is_Number:
                if abs(a) == 1: one = True                                  # 1 + x : x has no unit
                else: flags.add("adds a bare number (a built-in unit scale)")
                continue
            f = lin(a, cons, flags)
            if first is None: first = f
            else: cons.append(add(f, first, -1))
        if one and first is not None: cons.append(first); return {}
        return first or {}
    if isinstance(e, FUN):
        cons.append(lin(e.args[0], cons, flags)); return {}
    if isinstance(e, (sp.Abs, sp.Max, sp.Min)):
        first = None
        for a in e.args:
            f = lin(a, cons, flags)
            if first is None: first = f
            else: cons.append(add(f, first, -1))
        return first or {}
    raise Skip(type(e).__name__)


class Solver:
    def __init__(self): self.piv = {}; self.order = {}

    def key(self, v): return self.order.setdefault(v, len(self.order))

    def reduce(self, co, rhs):
        co = dict(co); rhs = list(rhs)
        while co:
            v = min(co, key=self.key)
            if v in self.piv:
                pc, pr = self.piv[v]; k = co[v]; co = add(co, pc, -k); rhs = [a - k * b for a, b in zip(rhs, pr)]
            else:
                k = co[v]; co = {u: c / k for u, c in co.items()}; rhs = [a / k for a in rhs]; return co, rhs, v
        return co, rhs, None

    def add_row(self, co, rhs):
        co, rhs, v = self.reduce(co, rhs)
        if v is None: return all(r == 0 for r in rhs)               # False = contradiction
        self.piv[v] = (co, rhs); return True

    def solution(self):
        det, free = {}, {}
        for v in sorted(self.piv, key=self.key, reverse=True):
            co, rhs = self.piv[v]
            co = dict(co); rhs = list(rhs)
            for u in [u for u in co if u != v]:
                if u in det: k = co.pop(u); rhs = [a - k * b for a, b in zip(rhs, det[u])]
            if len(co) == 1: det[v] = tuple(rhs)
            else: free[v] = [u for u in co if u != v]
        return det, free


def run(write=True):
    m = fm.Math(); S = Solver(); bad, skipped, scaled = [], [], {}
    for v, dims in SEED.items():
        if v in m.by_var: S.add_row({v: Fr(1)}, dims)
    for name, r in m.F.items():
        cons, flags = [], set()
        try:
            fl = lin(r["eq"].lhs, cons, flags); fr = lin(r["eq"].rhs, cons, flags)
        except Skip as e: skipped.append((name, str(e))); continue
        rows = [(add(fl, fr, -1), (Fr(0),) * 7)] + [(c, (Fr(0),) * 7) for c in cons]
        trial = Solver(); trial.piv = dict(S.piv); trial.order = dict(S.order); ok = True
        for co, rhs in rows:
            if not co: continue
            if not trial.add_row(co, rhs): ok = False; break
        if ok: S.piv, S.order = trial.piv, trial.order
        else: bad.append(name)
        if flags: scaled[name] = sorted(flags)
    det, free = S.solution()
    allv = sorted(m.by_var)
    table = {v: {"dims": [str(x) for x in det[v]], "text": U.dim_text(tuple(det[v])), "how": "given" if v in SEED else "worked out"} for v in allv if v in det}
    unknown = [v for v in allv if v not in det]
    if write:
        json.dump({"variables": table, "unknown": unknown, "does_not_balance": bad, "not_analysed": skipped, "built_in_scale": scaled}, open(os.path.join(fm.STORE, "units.json"), "w"), indent=1)
    return {"variables": len(allv), "known": len(table), "given": sum(1 for v in table if table[v]["how"] == "given"), "unknown": len(unknown), "does_not_balance": bad, "not_analysed": skipped, "built_in_scale": len(scaled)}, table, unknown, scaled


if __name__ == "__main__":
    res, table, unknown, scaled = run()
    print(json.dumps({k: v for k, v in res.items()}, indent=1)); print("unknown sample:", unknown[:60])
