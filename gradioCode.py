"""
Gradio playground for Hyper-Skin evaluation
==========================================

• Re-implements the Jupyter notebook you supplied, but wraps every step in an
  interactive UI built with Gradio.

• The code is **copy-paste ready** – just save it as, for example,
  `playground.py` and run

        python playground.py

  (Make sure you have already installed gradio:  `pip install gradio`)

• Except for the UI wrappers nothing was changed: the same dataset class,
  model definitions, helpers.metrics module, etc. are imported exactly as in
  your notebook, so the file/folder structure on disk must be untouched.

• By design the UI has two “pages”

    1.  Setup page – lets the user point to `data_dir` and `project_dir`
        (default paths are pre-filled) and then press
        “Start Playground”.  While the test set is evaluated a progress bar
        is shown.

    2.  Playground page – appears automatically when the evaluation is
        finished, shows the global scores at the top and three tabs:

        • Visual Comparison  
        • Spectral Profile  
        • Overall Analysis

  The behaviour of each tab is the same as in the notebook, with the extra
  widgets you asked for (sample index slider, RGB wavelength sliders, patch
  size slider, …).

• All heavy objects (dataset, results dictionary, SSIM/SAM arrays, …) are
  stored in a gr.State() variable so that every tab can access them without
  recomputation.

---------------------------------------------------------------------------
"""

# ----- Standard & 3rd-party imports --------------------------------------
from __future__ import absolute_import, division, print_function

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import cm
import cv2
import torch
from torchvision import transforms
import gradio as gr

# -------------------------------------------------------------------------
# --- Add project root to PYTHONPATH exactly as in the notebook -----------
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Local modules (unchanged) -------------------------------------------
from hsiData import HyperSkinData
from helpers import metrics
from hsiData.models.reconstruction import MST_Plus_Plus, HSCNN_Plus, hrnet


# ========================================================================
#                           helper functions
# ========================================================================

BAND_WAVELENGTHS = np.arange(400, 710, 10)          # 31 bands (unchanged)

def create_rgb_composite(hsi_cube: np.ndarray,
                         wavelengths: np.ndarray,
                         r_nm: int,
                         g_nm: int,
                         b_nm: int) -> np.ndarray:
    """
    Build an (HxWx3) pseudo-RGB image selecting three bands whose wavelength
    is *closest* to r_nm / g_nm / b_nm.  Data are stretched band-wise to the
    [0,1] range.
    """
    rgb_indices = [int(np.argmin(np.abs(wavelengths - wl)))
                   for wl in (r_nm, g_nm, b_nm)]
    rgb_bands = hsi_cube[rgb_indices, :, :].copy()

    for i in range(3):
        mi, ma = rgb_bands[i].min(), rgb_bands[i].max()
        if ma > mi:                                            # avoid /0
            rgb_bands[i] = (rgb_bands[i] - mi) / (ma - mi)

    return np.transpose(rgb_bands, (1, 2, 0))                 # H,W,3


def np_to_uint8_img(arr: np.ndarray) -> np.ndarray:
    """Utility – convert floating [0,1] or [0,max] array to uint8 RGB/BW"""
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def figure_to_numpy(fig: Figure) -> np.ndarray:
    """
    Convert a Matplotlib figure to an (H, W, 3) uint8 RGB array that Gradio
    can display.  Compatible with recent Matplotlib where
    `tostring_rgb()` was removed.
    """
    canvas = FigureCanvas(fig)          # attach a renderer
    canvas.draw()                       # draw the figure

    # Grab the RGBA buffer & reshape -------------------------------------------------
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    rgba = buf.reshape((h, w, 4))       # (H, W, 4)

    rgb = rgba[..., :3].copy()          # drop alpha
    plt.close(fig)
    return rgb


# ========================================================================
#                     heavy-lifting: evaluation routine
# ========================================================================

def evaluate_testset(data_dir: str,
                     project_dir: str,
                     progress=gr.Progress()) -> tuple:
    """
    1. Rebuilds exactly what your notebook cells 2-5 do
    2. Returns everything needed by the UI
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- paths -----------------------------------------------------------
    rgbvis_dir    = os.path.join(data_dir,   'Hyper-Skin(RGB, VIS)')
    pretrained_dir = os.path.join(project_dir, 'Models', 'RGB-VIS')

    rgb_dir  = os.path.join(rgbvis_dir, 'test', 'RGB')
    vis_dir  = os.path.join(rgbvis_dir, 'test', 'VIS')

    # ---------- dataset --------------------------------------------------
    test_transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = HyperSkinData.Load(hsi_dir=vis_dir,
                                      rgb_dir=rgb_dir,
                                      transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=(device == 'cuda'))
    # ---------- model ----------------------------------------------------
    model = MST_Plus_Plus.MST_Plus_Plus(in_channels=3,
                                        out_channels=31,
                                        n_feat=31,
                                        stage=3)
    chkpt_path = os.path.join(pretrained_dir, 'MSTPP-RGBVIS.pt')
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # ---------- evaluation ----------------------------------------------
    results = {"file": [], "pred": [],
               "ssim_score": [], "ssim_map": [],
               "sam_score": [],  "sam_map": []}

    metrics.ssim_fn.to(device)
    n_tot = len(test_loader)

    for k, data in enumerate(test_loader, start=1):
        progress(k / n_tot, desc=f"Evaluating {k}/{n_tot}")
        x, y = data
        x, y = x.float().to(device), y.float().to(device)
        with torch.no_grad():
            pred = model(x)

        ssim_score, ssim_map = metrics.ssim_fn(pred, y)
        sam_score,  sam_map  = metrics.sam_fn (pred, y)

        # --- gather outputs ---------------------------------------------
        results["file"].append(
            os.path.basename(test_dataset.rgb_files[k-1]).split('.')[0])
        results["pred"].append(pred.cpu().numpy())
        results["ssim_score"].append(ssim_score.cpu().numpy())
        results["ssim_map"].append(ssim_map.cpu().numpy())
        results["sam_score"].append(sam_score.cpu().numpy())
        results["sam_map"].append(sam_map.cpu().numpy())

    # lists -> np.array for convenience
    results["ssim_score"] = np.array(results["ssim_score"]).squeeze()
    results["sam_score"]  = np.array(results["sam_score"]).squeeze()

    # -------- summary string --------------------------------------------
    ssim_mu, ssim_sd = results["ssim_score"].mean(), results["ssim_score"].std()
    sam_mu,  sam_sd  = results["sam_score"].mean(),  results["sam_score"].std()

    summary = (f"SSIM: {ssim_mu:.4f} ± {ssim_sd:.4f} (higher is better)\n"
               f"SAM:  {sam_mu:.4f} ± {sam_sd:.4f} (lower is better)")

    return summary, results, test_dataset


# ========================================================================
#             Visual-comparison tab (all rendering helpers)
# ========================================================================

def visual_comparison(sample_idx, red_nm, green_nm, blue_nm,
                      results, dataset):
    # ── load tensors ----------------------------------------------------
    rgb_tensor, hsi_gt_tensor = dataset[sample_idx]
    hsi_gt   = hsi_gt_tensor.numpy()
    hsi_pred = np.array(results["pred"]).squeeze()[sample_idx]

    # ── composites ------------------------------------------------------
    gt_vis_rgb   = create_rgb_composite(hsi_gt,   BAND_WAVELENGTHS,
                                        red_nm, green_nm, blue_nm)
    pred_vis_rgb = create_rgb_composite(hsi_pred, BAND_WAVELENGTHS,
                                        red_nm, green_nm, blue_nm)

    # ── error maps (float) ---------------------------------------------
    mae_map = np.mean(np.abs(gt_vis_rgb - pred_vis_rgb), axis=2)
    sam_map_flat = results["sam_map"][sample_idx].squeeze()
    H, W = mae_map.shape
    sam_map = sam_map_flat.reshape(H, W)

    # ── colour-map helper ----------------------------------------------
    def apply_cmap(arr, cmap_name):
        arr = np.nan_to_num(arr)                # avoid NaNs
        rng = arr.max() - arr.min()
        norm = (arr - arr.min()) / (rng + 1e-8)
        rgba = cm.get_cmap(cmap_name)(norm)     # (H,W,4) float
        return (rgba[..., :3] * 255).astype(np.uint8)

    mae_img = apply_cmap(mae_map, 'hot')        # black->red->white
    sam_img = apply_cmap(sam_map, 'viridis')    # purple->yellow

    # ── RGB images ------------------------------------------------------
    in_rgb = (rgb_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gt_img = (gt_vis_rgb * 255).astype(np.uint8)
    pr_img = (pred_vis_rgb * 255).astype(np.uint8)

    return gt_img, pr_img, mae_img, sam_img


def visual_comparison_band(sample_idx: int,
                           band_nm: int,
                           results: dict,
                           dataset):
    """
    Shows one single wavelength (chosen by the user) instead of an RGB
    composite.

    Returns four images:
        1. Ground–truth band            (8-bit gray)
        2. Predicted band               (8-bit gray)
        3. |GT – PR| (Mean Absolute Error)   + colour-bar
        4. Per-pixel SAM map                  + colour-bar
    """
    # ── fetch tensors ----------------------------------------------------
    rgb_tensor, hsi_gt_tensor = dataset[sample_idx]
    hsi_gt   = hsi_gt_tensor.numpy()                       # (31,H,W)
    hsi_pred = np.array(results["pred"]).squeeze()[sample_idx]

    # ── pick the band ----------------------------------------------------
    band_idx = int(np.argmin(np.abs(BAND_WAVELENGTHS - band_nm)))
    gt_band  = hsi_gt [band_idx]                           # (H,W)
    pr_band  = hsi_pred[band_idx]

    # ── utility – float ➜ uint8 gray ------------------------------------
    def to_uint8(img):
        mi, ma = img.min(), img.max()
        if ma > mi:
            img = (img - mi) / (ma - mi)
        else:                                              # constant image
            img = np.zeros_like(img)
        return (img * 255).astype(np.uint8)

    gt_img = to_uint8(gt_band)           # (H,W) uint8
    pr_img = to_uint8(pr_band)

    # ── MAE map ----------------------------------------------------------
    mae_map = np.abs(gt_band - pr_band)
    fig_mae, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(mae_map, cmap='hot')
    ax.set_title('|GT – PR|')
    ax.axis('off')
    fig_mae.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    mae_img = figure_to_numpy(fig_mae)

    # ── SAM map (already pre-computed on full spectra) -------------------
    sam_map = results["sam_map"][sample_idx].squeeze()     # (H*W,)
    H, W = gt_band.shape
    sam_map = sam_map.reshape(H, W)

    fig_sam, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(sam_map, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Per-pixel SAM')
    ax.axis('off')
    fig_sam.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    sam_img = figure_to_numpy(fig_sam)

    return gt_img, pr_img, mae_img, sam_img

# ========================================================================
#          Spectral profile tab – patch selection & plotting
# ========================================================================

def spectral_profile(sample_idx: int,
                     top: int, left: int, size: int,
                     results: dict, dataset) -> tuple:
    """
    Returns:
        • VIS composite with a red rectangle marking the selected patch
        • spectral-profile plot (median + IQR)
    """
    # -------- tensors ---------------------------------------------------
    rgb_tensor, hsi_gt_tensor = dataset[sample_idx]
    hsi_gt   = hsi_gt_tensor.numpy()
    hsi_pred = np.array(results["pred"]).squeeze()[sample_idx]

    # -------- build the VIS composite (default RGB mapping) -------------
    vis_rgb = create_rgb_composite(hsi_pred, BAND_WAVELENGTHS, 610, 550, 460)
    vis_img = (vis_rgb * 255).astype(np.uint8).copy()

    # -------- draw patch rectangle --------------------------------------
    import cv2
    pt1 = (left, top)
    pt2 = (left + size, top + size)
    cv2.rectangle(vis_img, pt1, pt2, color=(255, 0, 0), thickness=2)  # red box

    # -------- extract patch & compute spectra ---------------------------
    yy = hsi_gt  [:, top:top+size, left:left+size].reshape(hsi_gt.shape[0], -1)
    pp = hsi_pred[:, top:top+size, left:left+size].reshape(hsi_gt.shape[0], -1)

    yy_med, yy_min, yy_max = np.median(yy, axis=1), \
                              np.percentile(yy, 25, axis=1), \
                              np.percentile(yy, 75, axis=1)

    pp_med, pp_min, pp_max = np.median(pp, axis=1), \
                              np.percentile(pp, 25, axis=1), \
                              np.percentile(pp, 75, axis=1)

    # -------- plot ------------------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    plt.plot(BAND_WAVELENGTHS, yy_med, 'g-', label='Ground Truth (median)')
    plt.fill_between(BAND_WAVELENGTHS, yy_min, yy_max, color='g', alpha=0.2)

    plt.plot(BAND_WAVELENGTHS, pp_med, 'b--', label='Prediction (median)')
    plt.fill_between(BAND_WAVELENGTHS, pp_min, pp_max, color='b', alpha=0.2)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f'Spectral profile – patch ({top}, {left}), {size}×{size}')
    plt.grid(True, ls=':')
    plt.legend()
    plot_img = figure_to_numpy(fig)

    return vis_img, plot_img


# ========================================================================
#                Overall analysis tab – static graphics
# ========================================================================

def overall_analysis_figures(results: dict) -> tuple:
    """Build histograms (SSIM & SAM) once"""
    ssim_scores = np.array(results["ssim_score"]).squeeze()
    sam_scores  = np.array(results["sam_score"]).squeeze()

    # --- SSIM histogram ---------------------------------------------------
    fig1 = plt.figure(figsize=(5,4))
    plt.hist(ssim_scores, bins=20, color='skyblue', edgecolor='k')
    plt.title('Distribution of SSIM scores')
    plt.xlabel('SSIM'); plt.ylabel('Frequency')
    plt.axvline(ssim_scores.mean(), color='r', ls='--',
                label=f'{ssim_scores.mean():.3f}')
    plt.legend()
    ssim_hist = figure_to_numpy(fig1)

    # --- SAM histogram ----------------------------------------------------
    fig2 = plt.figure(figsize=(5,4))
    plt.hist(sam_scores, bins=20, color='salmon', edgecolor='k')
    plt.title('Distribution of SAM scores')
    plt.xlabel('SAM'); plt.ylabel('Frequency')
    plt.axvline(sam_scores.mean(), color='r', ls='--',
                label=f'{sam_scores.mean():.3f}')
    plt.legend()
    sam_hist = figure_to_numpy(fig2)

    return ssim_hist, sam_hist


def ssim_band_grid(sample_idx: int, results: dict) -> np.ndarray:
    """
    exactly what notebook cell 11 did – grid of per-band SSIM maps
    (here rendered as a single image so that gr.Image can display it)
    """
    ssim_maps = np.array(results["ssim_map"]).squeeze()
    band_31   = BAND_WAVELENGTHS
    one = ssim_maps[sample_idx]                     # shape (31,H,W)

    # create a giant canvas ------------------------------------------------
    rows, cols = 4, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    for i in range(rows*cols-1):                    # last cell left blank
        if i >= one.shape[0]:
            axes[i].axis('off')
            continue
        im = axes[i].imshow(one[i], vmin=0, vmax=1, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'{band_31[i]} nm, mean={one[i].mean():.3f}')
    fig.colorbar(im, ax=axes.tolist(), pad=0.01)
    grid_img = figure_to_numpy(fig)
    return grid_img


def sam_maps_grid(results: dict) -> np.ndarray:
    """grid of SAM error maps across all test samples (notebook cell 12)"""
    sam_map = np.array(results["sam_map"]).squeeze()   # (N,H*W)
    n_imgs  = sam_map.shape[0]

    rows = int(np.ceil(n_imgs/6))
    fig, axes = plt.subplots(rows, 6, figsize=(18, 3*rows))
    axes = axes.flatten()
    for k, (img_flat, ax) in enumerate(zip(sam_map, axes)):
        sam_img = img_flat.reshape(-1, int(np.sqrt(img_flat.size)))
        im = ax.imshow(sam_img, vmin=0, vmax=1, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Sample {k}, μ={sam_img.mean():.3f}')
    for ax in axes[n_imgs:]:
        ax.axis('off')
    fig.colorbar(im, ax=axes.tolist(), pad=0.01)
    grid = figure_to_numpy(fig)
    return grid


# ========================================================================
#                         Gradio user-interface
# ========================================================================
DEFAULT_DATA_DIR    = 'D:/HyperSkin'
DEFAULT_PROJECT_DIR = ('C:/Users/user/Documents/StartMenuCustomFolders/'
                       'Research Work Stuff/Hyper-Skin-2023')

with gr.Blocks(title="Hyper-Skin – Interactive Evaluation") as demo:

    # States to share data between tabs -----------------------------------
    st_results  = gr.State()
    st_dataset  = gr.State()
    st_summary  = gr.State()

    # --------------------------------------------------------------------
    #                            PAGE 1 – Setup
    # --------------------------------------------------------------------
    with gr.Tab("Setup"):
        gr.Markdown("## Evaluation setup")
        gr.Markdown("Provide the root folders below (defaults are pre-filled) "
                    "and press **Start Playground**.  A progress bar will show "
                    "while the test set is evaluated.")
        data_dir_in  = gr.Textbox(value=DEFAULT_DATA_DIR, label="data_dir")
        proj_dir_in  = gr.Textbox(value=DEFAULT_PROJECT_DIR, label="project_dir")
        start_button = gr.Button("Start Playground")
        progress_box = gr.Markdown("")
        summary_box  = gr.Markdown("")              # will display SSIM / SAM

    # --------------------------------------------------------------------
    #                         PAGE 2 – Playground
    # --------------------------------------------------------------------
    with gr.Tab("Playground") as playground_tab:
        gr.Markdown("### Overall model performance")
        overall_summary = gr.Markdown(value="Run evaluation first…")

        # ---------------- Tabs inside playground -------------------------
        with gr.Tabs():

            # ============== 1) Visual comparison =========================
            with gr.Tab("Visual Comparison"):
                sample_slider = gr.Slider(label="Sample index",
                                        minimum=0, value=0, step=1)
                band_slider   = gr.Slider(400, 700, step=10, value=550,
                                        label="Band Selection (nm)")
                analyze_btn   = gr.Button("Analyze")

                with gr.Row(equal_height=True):
                    gt_img   = gr.Image(label="Ground-truth (single band)")
                    pred_img = gr.Image(label="Predicted (single band)")
                    mae_img  = gr.Image(label="Mean Absolute Error")
                    sam_img  = gr.Image(label="Per-pixel SAM")

            # ============== 2) Spectral profile ==========================
            with gr.Tab("Spectral Profile"):
                sp_sample_slider = gr.Slider(label="Sample index",
                                            minimum=0, maximum=1, value=0, step=1)
                patch_top   = gr.Slider(0, 1024, value=250, label="Patch top (row)")
                patch_left  = gr.Slider(0, 1024, value=500, label="Patch left (col)")
                patch_size  = gr.Slider(10, 200, value=50, step=10,
                                        label="Patch size (pixels)")
                plot_btn = gr.Button("Plot spectral profile")

                with gr.Row(equal_height=True):
                    vis_image = gr.Image(label="Reconstructed VIS (RGB composite)")
                    sp_plot   = gr.Image(label="Spectral profile")

            # ============== 3) Overall analysis ==========================
            with gr.Tab("Overall Analysis"):
                # 1st row ─ summary text
                oa_summary_md = gr.Markdown("Run evaluation first…")

                # 2nd row ─ histograms side-by-side
                with gr.Row(equal_height=True):
                    ssim_hist_img = gr.Image(label="Distribution of SSIM")
                    sam_hist_img  = gr.Image(label="Distribution of SAM")

                # 3rd row ─ left = slider+button+grid, right = SAM maps grid
                with gr.Row(equal_height=True):
                    with gr.Column():
                        oa_sample_slider = gr.Slider(
                            label="Sample index (for band SSIM)",
                            minimum=0, maximum=1, value=0, step=1
                        )
                        ssim_band_btn = gr.Button("Show per-band SSIM")
                        band_grid_img = gr.Image(label="SSIM per spectral band")
                    sam_grid_img = gr.Image(label="SAM maps across test set")


    # --------------------------------------------------------------------
    #                                Events
    # --------------------------------------------------------------------
    # ---- Run evaluation -------------------------------------------------
    def _run(start_btn, data_dir, project_dir, prog=gr.Progress()):
        summary, results, dataset = evaluate_testset(data_dir, project_dir, prog)
        n_samples = len(results["file"])

        # build static figs for overall analysis now (to avoid latency later)
        ssim_hist, sam_hist = overall_analysis_figures(results)
        sam_grid            = sam_maps_grid(results)

        # sliders need updated ranges
        slider_update   = gr.update(maximum=n_samples - 1, value=0)  # three sliders
        ready_update    = gr.update(value="Ready")                   # header
        setup_summary   = gr.update(value="Evaluation complete.")    # on Setup tab
        oa_summary_upd  = gr.update(value=summary)                   # OA tab text

        return (
            summary, results, dataset,          # state objects
            setup_summary,                      # summary_box  (Setup tab)
            ready_update,                       # overall_summary (header)
            oa_summary_upd,                     # NEW – oa_summary_md  (OA tab)

            slider_update, slider_update, slider_update,
            ssim_hist, sam_hist, sam_grid
        )

    start_button.click(
        _run,
        inputs=[start_button, data_dir_in, proj_dir_in],
        outputs=[
            st_summary, st_results, st_dataset,
            summary_box,          # Setup tab
            overall_summary,      # header ("Ready")
            oa_summary_md,        # NEW – Overall Analysis summary

            sample_slider, sp_sample_slider, oa_sample_slider,
            ssim_hist_img, sam_hist_img, sam_grid_img
        ]
    )

    # ---- Visual comparison ---------------------------------------------
    analyze_btn.click(
        visual_comparison_band,
        inputs=[sample_slider, band_slider, st_results, st_dataset],
        outputs=[gt_img, pred_img, mae_img, sam_img]
    )

    # ---- Spectral profile ----------------------------------------------
    plot_btn.click(
        spectral_profile,
        inputs=[sp_sample_slider, patch_top, patch_left, patch_size,
                st_results, st_dataset],
        outputs=[vis_image, sp_plot]
    )

    # ---- Per-band SSIM grid --------------------------------------------
    ssim_band_btn.click(
        ssim_band_grid,
        inputs=[oa_sample_slider, st_results],
        outputs=[band_grid_img]
    )

# ========================================================================
#                             launch
# ========================================================================
if __name__ == "__main__":
    demo.launch()

