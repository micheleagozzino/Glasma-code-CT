import numpy as np
import glob
import os
import sys
import re
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# in questa nuova versione "definitiva", cambiamo il modo di calcolare il v2 dei quark, offrendo la possibilità di considerare solo i q e non i qbar, oppure di considerare entrambi. Inoltre cambia il modo in cui, a partire da VnDelta e VnDelta_ref_ref, a seguito di opportuni consulti della teoria del v2.
# v2 = <VnDelta> / sqrt(<VnDelta_ref_ref>) ---> rapporto delle medie e NON media dei rapporti, come fatto nel precedente codice.
# Nota bene che il modo di calcolare i coefficienti VnDelta è lo stesso, ma non andremo più a scartare gli eventi con VnDelta_ref_ref < 0, ma li considereremo comunque. Nel caso in cui <VnDelta_ref_ref> < 0, non si potrà calcolare il v2, ma ciò significa solo che non siamo arrivati a convergenza statistica, e che quindi va fatto un conto con più eventi o particelle.
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- indice antiquark si/no ---
solo_quark = True  # se True, calcola solo per charm; se False, calcola anche per anticharm
tag = "solocharm" if solo_quark else "wanticharm"   # etichetta usata nei nomi dei file--> cambiala se stai facendo i bottom b

# --- setup cartella dati ---
percorso_file = os.path.abspath(__file__)
data_dir =  os.path.dirname(percorso_file)

# cartella di output per i file del v2 degli heavy quarks
out_dir = os.path.join(data_dir, "v2_hq")
os.makedirs(out_dir, exist_ok=True)

# --- kT binning ---
delta_kT = 0.5
kT_bins = np.arange(0.0, 7.0 + delta_kT, delta_kT)
kT_centers = 0.5 * (kT_bins[:-1] + kT_bins[1:])

# --- reference range for 2PC ---
pref_min, pref_max = 1.0, 6.0   # GeV

# --------- utilities ----------
def extract_number(f):
    m = re.search(r"_(\d+)\.csv$", os.path.basename(f))
    return int(m.group(1)) if m else -1

def formatta_valore_errore(val, err):
    """Restituisce '(val ± err)×10^k' in formato mathtext, con esponente comune."""
    if not (np.isfinite(val) and np.isfinite(err)):
        return f"{val:.2e} ± {err:.2e}"
    scala = max(abs(val), abs(err))
    if scala == 0:
        return r"$0 \pm 0$"
    k = int(np.floor(np.log10(scala)))
    return rf"$({val / 10**k:.2f} \pm {err / 10**k:.2f})\times 10^{{{k}}}$"

# --------- utilities for v2 calculation ----------
def calculate_VnDelta_event(kx, ky, w, kT_bins, n=2, pref_min=1.0, pref_max=6.0):
    """
    Compute VnDelta coefficients (pT and also ref-ref) for a single event, excluding self-correlations for each particle.
    Inputs:
      kx, ky : arrays of momenta
      w      : weights (here = 1 per particle)
      kT_bins: bin edges in kT
      n      : harmonic order (default 2)
    Returns:
      VnDelta_refref, VnDelta (per-bin)
    """
    kT = np.sqrt(kx**2 + ky**2)
    phi = np.arctan2(ky, kx)

    nbins = len(kT_bins) - 1
    Qn_bins = np.zeros(nbins, dtype=complex)
    N_bins  = np.zeros(nbins)

    # build Qn for each kT bin
    for i, (k_lo, k_hi) in enumerate(zip(kT_bins[:-1], kT_bins[1:])):
        mask = (kT >= k_lo) & (kT < k_hi)
        if np.any(mask):
            w_sel = w[mask]
            phi_sel = phi[mask]
            Qn_bins[i] = np.sum(w_sel * np.exp(1j * n * phi_sel))
            N_bins[i] = np.sum(w_sel)

    VnDelta = np.full(nbins, np.nan)

    for i in range(nbins):
        if N_bins[i] > 0:
           
            mask_ref = (kT >= pref_min) & (kT <= pref_max) 

            if not np.any(mask_ref):
                continue  # no reference particles left after excluding bin i

            w_ref = w[mask_ref]
            phi_ref = phi[mask_ref]
            Qn_ref = np.sum(w_ref * np.exp(1j * n * phi_ref))
            N_ref  = np.sum(w_ref)

            if N_ref > 0:

                k_bin_center = 0.5 * (kT_bins[i] + kT_bins[i+1])

                if pref_min <= k_bin_center <= pref_max:
                # bin is inside the reference window → exclude self-correlations
                    VnDelta[i] = (np.real(Qn_bins[i] * np.conj(Qn_ref)) - N_bins[i]) / (N_bins[i] * (N_ref - 1))

                else:
            # bin is outside reference window → normal calculation
                    VnDelta[i] = np.real(Qn_bins[i] * np.conj(Qn_ref)) / (N_bins[i] * N_ref)

    # --- global ref-ref (full pref range, no exclusion) ---
    mask_ref_global = (kT >= pref_min) & (kT <= pref_max)
    # if not np.any(mask_ref_global):
    #     return np.full(nbins, np.nan), np.nan
    if not np.any(mask_ref_global):
        # nessuna particella di riferimento: refref non definito, stesso ordine del return finale
        return np.nan, VnDelta

    w_ref_g = w[mask_ref_global]
    phi_ref_g = phi[mask_ref_global]
    Qn_ref_g = np.sum(w_ref_g * np.exp(1j * n * phi_ref_g))
    N_ref_g  = np.sum(w_ref_g)
    VnDelta_refref = (np.real(Qn_ref_g * np.conj(Qn_ref_g))-N_ref_g) / (N_ref_g * (N_ref_g-1))

    #print(f"Evento: VnDelta_refref = {VnDelta_refref:.6f}, N_ref_g = {N_ref_g}")
    

    return VnDelta_refref, VnDelta

#=================================MAIN====================================================================
#--------------------
files = sorted(glob.glob(os.path.join(data_dir, "HQinitializations/XandP_tfin_*.csv")), key=extract_number)
if not files:
    print("No files found matching XandP_tfin_*.csv")
    sys.exit()

VnDelta_refref_all = [] #creo un array su tutti gli eventi per raccogliere i valori di VnDelta_refref e VnDelta (quest'ultimo è un array di array, con un array per ogni evento, contenente i valori di VnDelta per ogni bin)
VnDelta_all = []    

for i, filename in enumerate(files):
    data = np.genfromtxt(filename, delimiter=',')
    
    if solo_quark:
        Nhalf = data.shape[0] // 2 # righe [0, Nhalf) = charm, [Nhalf, NHQ) = anticharm
        kx = data[:Nhalf,2]
        ky = data[:Nhalf,3]
    else:
        kx = data[:,2]
        ky = data[:,3]

    w  = np.ones_like(kx)   # one weight per particle (just counting)

    V2_refref, VnDelta_evt = calculate_VnDelta_event(
        kx, ky, w, kT_bins, n=2, pref_min=pref_min, pref_max=pref_max
    )
    VnDelta_refref_all.append(V2_refref)
    VnDelta_all.append(VnDelta_evt)
        
    #print(f"Processed {i}/{len(files)}: {filename}")

# ---------- media sugli eventi di VnDelta_refref -----------------------------------------
VnDelta_refref_all  = np.array(VnDelta_refref_all)
N_ev_rr     = np.sum(~np.isnan(VnDelta_refref_all))
VnDelta_refref_mean = np.nanmean(VnDelta_refref_all)
VnDelta_refref_err  = np.nanstd(VnDelta_refref_all, ddof=1) / np.sqrt(N_ev_rr)
print(f"<VnDelta_refref> = {VnDelta_refref_mean:.6e} +- {VnDelta_refref_err:.6e}  (N_eventi = {N_ev_rr})")

# media ed errore salvati a parte: servono all'etichetta del grafico di confronto
np.savetxt(os.path.join(out_dir, f"VnDelta_refref_media_{tag}_{len(files)}events_deltapt05.txt"),
           [[VnDelta_refref_mean, VnDelta_refref_err, N_ev_rr]], fmt="%.8e",
           header="<VnDelta_refref>  stat.err  N_eventi")
etichetta_refref = r"$\langle V_{2\Delta}^{ref,ref}\rangle$ = " + \
                   formatta_valore_errore(VnDelta_refref_mean, VnDelta_refref_err)
# VnDelta_ref_ref stampati evento per evento in un file di testo (per una contezza di quanti eventi hanno VnDelta_refref < 0v molto utile nel caso senza campi, e ci ha aiutati a capire che non potevamo scartare così tanti eventi.)
if solo_quark:
    print(f"Results for charm only (anticharm excluded).")
    np.savetxt(os.path.join(out_dir, f"VnDelta_refref_per_evento_{tag}_{len(files)}events_deltapt05.txt"),
           VnDelta_refref_all, fmt="%.8e",
           header=f"VnDelta_refref per evento, pref=[{pref_min},{pref_max}] GeV; "
                  f"media = {VnDelta_refref_mean:.8e} +- {VnDelta_refref_err:.8e}")
else:
    print(f"Results for charm + anticharm.")
    np.savetxt(os.path.join(out_dir, f"VnDelta_refref_per_evento_{tag}_{len(files)}events_deltapt05.txt"),
           VnDelta_refref_all, fmt="%.8e",
           header=f"VnDelta_refref per evento, pref=[{pref_min},{pref_max}] GeV; "
                  f"media = {VnDelta_refref_mean:.8e} +- {VnDelta_refref_err:.8e}")

# ---------- media sugli eventi di VnDelta(pT) --------------------------------------------
VnDelta_all = np.array(VnDelta_all)                       # shape (N_eventi, N_bin)
N_ev_bin    = np.sum(~np.isnan(VnDelta_all), axis=0)      # eventi validi in ogni bin
with np.errstate(invalid="ignore", divide="ignore"):
    VnDelta_mean = np.nanmean(VnDelta_all, axis=0)
    VnDelta_err  = np.nanstd(VnDelta_all, axis=0, ddof=1) / np.sqrt(N_ev_bin)
    VnDelta_sig  = VnDelta_mean / VnDelta_err             # distanza da zero in sigma

out_vd = np.column_stack((kT_centers, VnDelta_mean, VnDelta_err, VnDelta_sig, N_ev_bin))
if solo_quark:
    np.savetxt(os.path.join(out_dir, f"VnDelta_vs_kT_{tag}_{len(files)}events_deltapt05.txt"), out_vd, fmt="%.8e",
               header=f"kT[GeV]  <VnDelta>  stat.err  <VnDelta>/err  N_eventi_bin   "
                      f"(pref = [{pref_min}, {pref_max}] GeV)")
else:
    np.savetxt(os.path.join(out_dir, f"VnDelta_vs_kT_{tag}_{len(files)}events_deltapt05.txt"), out_vd, fmt="%.8e",
               header=f"kT[GeV]  <VnDelta>  stat.err  <VnDelta>/err  N_eventi_bin   "
                      f"(pref = [{pref_min}, {pref_max}] GeV)")

# ---------- v2(pT) = <VnDelta(pT)> / sqrt(<VnDelta_refref>) --------------------------------
# X = <VnDelta>(pT), Y = <VnDelta_refref>. Propagazione degli errori:
#   dv2^2 = dX^2/Y + X^2 dY^2/(4 Y^3) - (X/Y^2) cov(X,Y)
# X e Y sono medie sugli stessi eventi, quindi correlate: cov(X,Y) = cov_eventi(VnDelta, refref)/N.
# Salvato anche l'errore senza covarianza (formula con X e Y indipendenti) per confronto.
X_v2, dX_v2 = VnDelta_mean, VnDelta_err
Y_v2, dY_v2 = VnDelta_refref_mean, VnDelta_refref_err

cov_XY = np.full(len(kT_centers), np.nan)
for b in range(len(kT_centers)):
    ok = ~np.isnan(VnDelta_all[:, b]) # eventi validi per entrambi
    n_ok = np.sum(ok)
    if n_ok > 1:
        cov_XY[b] = np.cov(VnDelta_all[ok, b], VnDelta_refref_all[ok], ddof=1)[0, 1] / n_ok

# v2 definito solo se <VnDelta_refref> è un numero finito e positivo.
# Il caso NaN si ha quando nessuna particella cade nella finestra di riferimento
# (NaN <= 0 è False, quindi il solo test "Y_v2 <= 0" non lo intercetterebbe).
v2_definito = bool(np.isfinite(Y_v2) and Y_v2 > 0)

if not v2_definito:
    # <VnDelta_refref> = <v2_ref^2> non può essere negativo: la media non è ancora convergente
    # (fluttuazioni statistiche dominanti). v2 non è definito: nessun file e nessun grafico.
    print("=" * 90)
    print(f"PROBLEMA DI CONVERGENZA STATISTICA: <VnDelta_refref> = {Y_v2:.3e} +- {dY_v2:.3e} <= 0 (o non definito)")
    print(f"(N_eventi = {N_ev_rr}). v2 = <VnDelta>/sqrt(<VnDelta_refref>) non è definito:")
    print("il grafico e il file del v2 (rapporto delle medie) NON vengono prodotti.")
    print("Servono più eventi (o più particelle di riferimento) perché la media converga.")
    print("=" * 90)
else:
    with np.errstate(invalid="ignore"):
        v2_rm       = X_v2 / np.sqrt(Y_v2)
        var_indip   = dX_v2**2 / Y_v2 + X_v2**2 * dY_v2**2 / (4 * Y_v2**3)
        v2_rm_err_indip = np.sqrt(var_indip)
        v2_rm_err   = np.sqrt(np.clip(var_indip - (X_v2 / Y_v2**2) * cov_XY, 0, None))
    if Y_v2 < 3 * dY_v2:
        print(f"ATTENZIONE: <VnDelta_refref> = {Y_v2:.3e} +- {dY_v2:.3e} è compatibile con zero "
              f"entro 3 sigma: v2 = X/sqrt(Y) e la propagazione lineare degli errori non sono affidabili.")

    out_v2rm = np.column_stack((kT_centers, v2_rm, v2_rm_err, v2_rm_err_indip, X_v2, dX_v2, cov_XY))
    np.savetxt(os.path.join(out_dir, f"v2_rapportoMedie_{tag}_{len(files)}events_deltapt05.txt"), out_v2rm, fmt="%.8e",
               header=(f"kT[GeV]  v2=<VnDelta>/sqrt(<VnDelta_refref>)  err(con cov)  err(senza cov)  "
                       f"<VnDelta>  err<VnDelta>  cov(<VnDelta>,<VnDelta_refref>)\n"
                       f"<VnDelta_refref> = {Y_v2:.8e} +- {dY_v2:.8e}   (pref = [{pref_min}, {pref_max}] GeV)"))


#======================================GRAFICI======================================================================

#1) grafico del v2 (grafico più importante) -- solo se il v2 è definito
if v2_definito:
    plt.figure()
    plt.errorbar(kT_centers, v2_rm, yerr=v2_rm_err, fmt="o", capsize=3, color="tab:green",
                    label=r"$v_2 = \langle V_{2\Delta}\rangle / \sqrt{\langle V_{2\Delta}^{ref,ref}\rangle}$")
    plt.axhline(0, color="black", ls="--", lw=1)
    plt.axvspan(pref_min, min(pref_max, kT_bins[-1]), color="gray", alpha=0.15,
                label="finestra di riferimento")
    plt.plot([], [], " ", label=etichetta_refref)   # voce di legenda con <VnDelta_refref>
    plt.xlim(kT_bins[0], kT_bins[-1])
    plt.xlabel(r"$k_T$ [GeV]")
    plt.ylabel(r"$v_2$")
    plt.title(f"{'solo charm' if solo_quark else 'charm + anticharm'}, {len(files)} eventi")
    plt.legend(fontsize=8); plt.grid(True)
    plt.tight_layout()
    file_v2rm = os.path.join(out_dir, f"v2_2pc_{tag}_{len(files)}events_deltapt05.png")
    plt.savefig(file_v2rm, dpi=300)
    plt.close()
    print(f"v2 (rapporto delle medie) salvato in {file_v2rm}")