"""
Utilitare pentru vizualizarea rezultatelor (curbe Pfa/Pd, hărți de detecție).

Stilul graficelor urmărește convențiile din articolele IEEE de referință
(De Maio 2009, Pauciullo 2018, Dănișor 2023) pentru a facilita compararea
vizuală.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Schemă de culori inspirată din articolele de referință
COLORS = {
    'SL-GLRT': '#0072BD',   # albastru
    'ML-GLRT-9': '#D95319',   # portocaliu
    'ML-GLRT-25': '#EDB120',  # galben/auriu
    'MIC': '#7E2F8E',         # mov
    'CAESAR-D': '#77AC30',    # verde
    'SqueeSAR-D': '#A2142F',  # roșu închis
}


def plot_pfa_curve(thresholds, pfa_dict, title=None, ax=None, ylim=None):
    """
    Plotează Pfa vs T pe scară log pentru mai multe detectoare.

    Parameters
    ----------
    thresholds : np.ndarray
    pfa_dict : dict {nume_detector: pfa_array}
    title : str, opțional
    ax : matplotlib axes, opțional
    ylim : tuple, opțional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    for name, pfa in pfa_dict.items():
        color = COLORS.get(name, None)
        ax.semilogy(thresholds, np.maximum(pfa, 1e-7),
                    label=name, color=color, linewidth=2)

    ax.set_xlabel(r'Pragul $T$', fontsize=12)
    ax.set_ylabel(r'Probabilitatea de alarmă falsă $P_{fa}$', fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    if title:
        ax.set_title(title, fontsize=13)
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, ax


def plot_pd_curve(snr_db_values, pd_dict, title=None, ax=None):
    """
    Plotează Pd vs SNR pentru mai multe detectoare.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    for name, pd in pd_dict.items():
        color = COLORS.get(name, None)
        ax.plot(snr_db_values, pd, label=name, color=color,
                linewidth=2, marker='o', markersize=4)

    ax.set_xlabel(r'SNR (dB)', fontsize=12)
    ax.set_ylabel(r'Probabilitatea de detecție $P_d$', fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    if title:
        ax.set_title(title, fontsize=13)
    return fig, ax


def plot_baselines(geom, title='Distribuția spațio-temporală a achizițiilor',
                   ax=None):
    """
    Plotează distribuția baseline-urilor perpendiculare vs temporale.
    Stilul corespunde cu Fig. 4 din Dănișor et al. 2023.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    bperp = geom.baselines_perp
    btemp_days = geom.baselines_temp * 365.25

    ax.scatter(btemp_days, bperp, marker='o', s=50,
               facecolors='none', edgecolors='b', linewidth=1.5)
    # Master cu un alt simbol
    master_idx = np.argmin(np.abs(bperp) + np.abs(btemp_days))
    ax.scatter(btemp_days[master_idx], bperp[master_idx],
               marker='*', s=200, color='yellow', edgecolor='black',
               linewidth=1.5, zorder=5, label='Master')

    ax.set_xlabel('Linia de bază temporală (zile)', fontsize=12)
    ax.set_ylabel('Linia de bază perpendiculară (m)', fontsize=12)
    ax.axhline(0, color='gray', alpha=0.5, linestyle=':')
    ax.axvline(0, color='gray', alpha=0.5, linestyle=':')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='best')
    return fig, ax


def plot_detection_map(results, background=None, title=None, ax=None,
                       color_by='v_est', vmin=None, vmax=None):
    """
    Plotează harta de detecție pe un fundal opțional.

    Parameters
    ----------
    results : dict (rezultat de la detect_stack)
    background : np.ndarray sau None
        Imagine de fundal (ex: media temporală a amplitudinilor).
    color_by : str, 'v_est' sau 's_est' sau 'statistic'
        Ce să folosească pentru colorarea PS detectați.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    detected = results['detected']

    # Fundal
    if background is not None:
        ax.imshow(background, cmap='gray', alpha=0.7)

    # Punctele detectate, colorate după parametrul ales
    yy, xx = np.where(detected)
    values = results[color_by][detected]

    if color_by == 'v_est':
        # cm/an pentru afișare
        values_display = values * 100
        cmap = 'RdBu_r'
        label = 'Viteză deformare (cm/an)'
        if vmin is None:
            vmin, vmax = -2, 2
    elif color_by == 's_est':
        values_display = values
        cmap = 'viridis'
        label = 'Elevație reziduală (m)'
        if vmin is None:
            vmin, vmax = -30, 30
    else:
        values_display = values
        cmap = 'hot'
        label = 'Statistica γ'
        if vmin is None:
            vmin, vmax = 0, 1

    sc = ax.scatter(xx, yy, c=values_display, cmap=cmap, s=4,
                    vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label=label, shrink=0.7)

    ax.set_xlabel('Range (pixeli)')
    ax.set_ylabel('Azimuth (pixeli)')
    if title is None:
        title = f'Dispersori detectați: {detected.sum()} pixeli'
    ax.set_title(title, fontsize=13)
    return fig, ax


def plot_scatter_estimates(results_dict, s_true=None, v_true=None,
                           title='Scatter plot estimări (s, v)'):
    """
    Compară estimările (s, v) pentru mai mulți detectori (similar Fig. 3
    din Dănișor et al. 2023).

    Parameters
    ----------
    results_dict : dict {nume_detector: (s_est_array, v_est_array)}
    s_true, v_true : valorile reale (dacă disponibile)
    """
    n_detectors = len(results_dict)
    fig, axes = plt.subplots(1, n_detectors, figsize=(6 * n_detectors, 5),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()

    for ax, (name, (s_est, v_est)) in zip(axes, results_dict.items()):
        ax.scatter(s_est, v_est * 100, s=8, alpha=0.5,
                   color=COLORS.get(name, 'b'))
        if s_true is not None and v_true is not None:
            ax.scatter(s_true, v_true * 100, s=200, marker='+',
                       color='red', linewidths=3, label='Valoarea reală')
            ax.legend()
        ax.set_xlabel('Elevație reziduală (m)', fontsize=11)
        ax.set_ylabel('Viteză LOS (cm/an)', fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig, axes
