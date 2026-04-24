# Authors:
# Maxime Rizzo
# NASA GSFC, 2026

# Contributors:
# David Morgan

#%%
import os
from pathlib import Path
import io
import re
from ast import literal_eval
import pandas as pd
import tarfile
import zipfile
import argparse
import html as html_module
import json
from astropy.coordinates import SkyCoord
from roman_opup_tools import roman_attitude
from roman_visit_viewer.roman_visit_viewer import VisitFileParser, plot_manager
import shutil
import matplotlib

# Columns to make first in the output CSV file (if available)
PRIORITY_COLUMNS = ['Visit_ID', 'SCI_ID', 'Visit_File_Name', 'RA_V1 [calc]', 'DEC_V1 [calc]', 'V3PA_V1 [calc]', 'RA_WFI_CEN [calc]', 'DEC_WFI_CEN [calc]', 'V3PA_WFI_CEN [calc]',
                    'Off-Normal_Roll', 'Off-Normal_Roll [calc]', 'Pitch [calc]',
                    'WFI_SCI_TABLE', 'READFRAMES', 'WFI_Optical_Element']
#%%

# ════════════════════════════════════════════════════════════════════════════
# PART 1: Quaternion-based WFI footprint precomputation
# Add to visit_parser.py
# ════════════════════════════════════════════════════════════════════════════

import numpy as np
import json

# Column names for the ECI→BCS quaternion in the OPUP table
QUAT_COLS = ['TAR_Q1_ECI2BCS', 'TAR_Q2_ECI2BCS', 'TAR_Q3_ECI2BCS', 'TAR_Q4_ECI2BCS']

def precompute_wfi_footprints(df):
    """
    For each unique visit (keyed by Visit_ID + SCI_ID), read the pointing
    quaternion from the TAR_Q*_ECI2BCS columns, convert to RA/Dec/PA via
    the same math as roman_visit_viewer.roman_attitude(), build the pysiaf
    attitude matrix, and extract the sky corners of all 18 WFI SCAs.

    Falls back to RA / DEC / Position_Angle columns when quaternion data
    is unavailable for a row.

    Returns
    -------
    dict
        JSON-serializable, keyed by a unique row key (Visit_ID or
        Visit_ID + SCI_ID), each value:
        {
            'ra': float,  'dec': float,  'pa': float,   # V1 boresight
            'ra_cen': float, 'dec_cen': float,           # WFI_CEN on sky
            'scas': {
                'WFI01_FULL': [[ra, dec], ...],           # closed polygon
                ...
            }
        }
    """
    try:
        import pysiaf
        import astropy.units as u
    except ImportError:
        print("  ⚠️  pysiaf not installed — skipping WFI footprint precomputation.")
        return {}

    RSIAF = pysiaf.Siaf('Roman')

    # ── Determine whether quaternion columns are available ──
    have_quat = all(c in df.columns for c in QUAT_COLS)
    have_radec = all(c in df.columns for c in ['RA', 'DEC', 'Position_Angle'])
    if not have_quat and not have_radec:
        print("  ⚠️  Neither quaternion nor RA/DEC/PA columns found — cannot compute footprints.")
        return {}

    if have_quat:
        print("  📐 Using TAR_Q*_ECI2BCS quaternion columns (exact pointing)")
    else:
        print("  📐 Quaternion columns not found — falling back to RA/DEC/Position_Angle")

    # ── Build a unique key per exposure row ──
    # Use Visit_ID + SCI_ID so every dither gets its own footprint
    key_cols = []
    if 'Visit_ID' in df.columns:
        key_cols.append('Visit_ID')
    if 'SCI_ID' in df.columns:
        key_cols.append('SCI_ID')
    if not key_cols:
        print("  ⚠️  No Visit_ID column — cannot key footprints.")
        return {}

    # Drop duplicate keys (same pointing doesn't need recomputing)
    pointing_cols = QUAT_COLS if have_quat else ['RA', 'DEC', 'Position_Angle']
    # Include all GW columns so guide star positions can be computed
    gw_cols = [c for c in df.columns if c.startswith('TRK_')]
    needed_cols = key_cols + pointing_cols + ['RA', 'DEC', 'Position_Angle'] + gw_cols
    needed_cols = [c for c in needed_cols if c in df.columns]
    # De-duplicate
    needed_cols = list(dict.fromkeys(needed_cols))
    unique_rows = df.drop_duplicates(subset=key_cols)[needed_cols].dropna(subset=pointing_cols)

    print(f"  🔭 Computing WFI footprints for {len(unique_rows)} unique exposures...")

    footprints = {}

    for _, row in unique_rows.iterrows():
        # Build unique key string
        row_key = '_'.join(str(row[c]) for c in key_cols)

        try:
            # ── Derive RA, Dec, V3PA ──
            if have_quat:
                q1 = float(row['TAR_Q1_ECI2BCS'])
                q2 = float(row['TAR_Q2_ECI2BCS'])
                q3 = float(row['TAR_Q3_ECI2BCS'])
                q4 = float(row['TAR_Q4_ECI2BCS'])
                ra_v1, dec_v1, v3pa = roman_attitude.quat_to_radec_pa(q1, q2, q3, q4)
            else:
                ra_v1  = float(row['RA'])
                dec_v1 = float(row['DEC'])
                v3pa   = float(row['Position_Angle'])

            # ── Build attitude matrix  (same as Exposure.plot()) ──
            att_mat = pysiaf.rotations.attitude_matrix(0, 0, ra_v1, dec_v1, v3pa)

            # ── WFI_CEN sky position (Aladin centering target) ──
            wfi_cen = RSIAF['WFI_CEN']
            wfi_cen.set_attitude_matrix(att_mat)
            ra_wfi, dec_wfi = pysiaf.rotations.tel_to_sky(
                att_mat, wfi_cen.V2Ref, wfi_cen.V3Ref
            )
            ra_cen  = float(ra_wfi.to(u.deg).value) if hasattr(ra_wfi, 'to') else float(ra_wfi)
            dec_cen = float(dec_wfi.to(u.deg).value) if hasattr(dec_wfi, 'to') else float(dec_wfi)

            visit_fp = {
                'ra':      round(ra_v1, 7),
                'dec':     round(dec_v1, 7),
                'pa':      round(v3pa, 4),
                'ra_cen':  round(ra_cen, 7),
                'dec_cen': round(dec_cen, 7),
                'scas':    {}
            }

            # ── Extract corners for each of the 18 SCAs ──
            for isca in range(1, 19):
                sca_name = f'WFI{isca:02d}_FULL'
                aper = RSIAF[sca_name]
                aper.set_attitude_matrix(att_mat)

                ra_corners = []
                dec_corners = []
                for iv in range(1, 5):
                    x_idl = getattr(aper, f'XIdlVert{iv}')
                    y_idl = getattr(aper, f'YIdlVert{iv}')
                    ra_sky, dec_sky = aper.idl_to_sky(x_idl, y_idl)
                    ra_corners.append(float(ra_sky))
                    dec_corners.append(float(dec_sky))

                # Close the polygon
                ra_corners.append(ra_corners[0])
                dec_corners.append(dec_corners[0])

                visit_fp['scas'][sca_name] = [
                    [round(r, 7), round(d, 7)]
                    for r, d in zip(ra_corners, dec_corners)
                ]
            # ── Extract CGI aperture corners ──
            try:
                cgi_aper = RSIAF['CGI_CEN']
                cgi_aper.set_attitude_matrix(att_mat)
                ra_corners = []
                dec_corners = []
                for iv in range(1, 5):
                    x_idl = getattr(cgi_aper, f'XIdlVert{iv}')
                    y_idl = getattr(cgi_aper, f'YIdlVert{iv}')
                    ra_sky, dec_sky = cgi_aper.idl_to_sky(x_idl, y_idl)
                    ra_corners.append(float(ra_sky))
                    dec_corners.append(float(dec_sky))
                ra_corners.append(ra_corners[0])
                dec_corners.append(dec_corners[0])
                visit_fp['cgi'] = [
                    [round(r, 7), round(d, 7)]
                    for r, d in zip(ra_corners, dec_corners)
                ]
            except Exception:
                pass  # CGI aperture not available in this SIAF version
            # ── 6. Extract guide window positions ──
            # Columns: TRK_USE_GWxx ("GUIDE" or "SKY_FIXED")
            #          TRK_H_GWxx, TRK_V_GWxx (FGS frame coords)
            guide_stars = []
            for isca in range(1, 19):
                use_col = f'TRK_USE_GW{isca:02d}'
                h_col   = f'TRK_H_GW{isca:02d}'
                v_col   = f'TRK_V_GW{isca:02d}'

                if not all(c in row.index for c in [use_col, h_col, v_col]):
                    continue

                mode_val = str(row.get(use_col, '')).strip().strip('"')
                if not mode_val or mode_val == 'nan':
                    continue

                try:
                    fgs_x = float(row[h_col])
                    fgs_y = float(row[v_col])
                except (ValueError, TypeError):
                    continue

                # Convert FGS frame → science frame → sky
                # (same as roman_visit_viewer.py Exposure.plot())
                scix = fgs_x + 2048
                sciy = 2048 - fgs_y

                sca_name = f'WFI{isca:02d}_FULL'
                aper = RSIAF[sca_name]
                aper.set_attitude_matrix(att_mat)
                gs_ra, gs_dec = aper.sci_to_sky(scix, sciy)

                guide_stars.append({
                    'sca': isca,
                    'ra': round(float(gs_ra), 7),
                    'dec': round(float(gs_dec), 7),
                    'mode': mode_val   # "GUIDE" or "SKY_FIXED"
                })

            visit_fp['guide_stars'] = guide_stars

            footprints[row_key] = visit_fp

        except Exception as e:
            print(f"    ⚠️  Footprint failed for {row_key}: {e}")
            continue

    print(f"  ✅ Computed footprints for {len(footprints)}/{len(unique_rows)} exposures")
    return footprints

def add_pointing_columns(df):
    """
    Compute RA, Dec, V3PA at V1 boresight and at WFI_CEN from the 
    TAR_Q*_ECI2BCS quaternion columns, and add them to the DataFrame.
    
    New columns added:
        RA [calc]              - V1 boresight RA (degrees)
        DEC [calc]             - V1 boresight Dec (degrees)
        V3PA [calc]            - V3 position angle at V1 (degrees)
        RA_WFI_CEN [calc]     - RA at WFI_CEN aperture (degrees)
        DEC_WFI_CEN [calc]    - Dec at WFI_CEN aperture (degrees)
        V3PA_WFI_CEN [calc]   - V3PA at WFI_CEN aperture (degrees)
    """
    have_quat = all(c in df.columns for c in QUAT_COLS)
    if not have_quat:
        print("  ⚠️  Quaternion columns not found — skipping pointing columns")
        return df

    try:
        import pysiaf
        import astropy.units as u
        RSIAF = pysiaf.Siaf('Roman')
    except ImportError:
        print("  ⚠️  pysiaf not installed — skipping pointing columns")
        return df

    ra_v1_list = []
    dec_v1_list = []
    v3pa_v1_list = []
    ra_wfi_list = []
    dec_wfi_list = []
    v3pa_wfi_list = []

    for _, row in df.iterrows():
        try:
            q1 = float(row['TAR_Q1_ECI2BCS'])
            q2 = float(row['TAR_Q2_ECI2BCS'])
            q3 = float(row['TAR_Q3_ECI2BCS'])
            q4 = float(row['TAR_Q4_ECI2BCS'])

            if any(np.isnan(v) for v in [q1, q2, q3, q4]):
                raise ValueError("NaN quaternion")

            # Quaternion → RA, Dec, V3PA at V1
            ra_v1, dec_v1, v3pa = roman_attitude.quat_to_radec_pa(q1, q2, q3, q4)

            # Build attitude matrix
            att_mat = pysiaf.rotations.attitude_matrix(0, 0, ra_v1, dec_v1, v3pa)

            # WFI_CEN sky position
            wfi_cen = RSIAF['WFI_CEN']
            ra_wfi, dec_wfi = pysiaf.rotations.tel_to_sky(
                att_mat, wfi_cen.V2Ref, wfi_cen.V3Ref
            )
            ra_wfi = float(ra_wfi.to(u.deg).value) if hasattr(ra_wfi, 'to') else float(ra_wfi)
            dec_wfi = float(dec_wfi.to(u.deg).value) if hasattr(dec_wfi, 'to') else float(dec_wfi)
            v3pa_wfi = pysiaf.rotations.posangle(att_mat, wfi_cen.V2Ref, wfi_cen.V3Ref)
            v3pa_wfi = float(v3pa_wfi.to(u.deg).value) if hasattr(v3pa_wfi, 'to') else float(v3pa_wfi)

            ra_v1_list.append(round(ra_v1, 6))
            dec_v1_list.append(round(dec_v1, 6))
            v3pa_v1_list.append(round(v3pa, 4))
            ra_wfi_list.append(round(ra_wfi, 6))
            dec_wfi_list.append(round(dec_wfi, 6))
            v3pa_wfi_list.append(round(v3pa_wfi, 4))

        except Exception:
            ra_v1_list.append(None)
            dec_v1_list.append(None)
            v3pa_v1_list.append(None)
            ra_wfi_list.append(None)
            dec_wfi_list.append(None)
            v3pa_wfi_list.append(None)

    df['RA_V1 [calc]'] = ra_v1_list
    df['DEC_V1 [calc]'] = dec_v1_list
    df['V3PA_V1 [calc]'] = v3pa_v1_list
    df['RA_WFI_CEN [calc]'] = ra_wfi_list
    df['DEC_WFI_CEN [calc]'] = dec_wfi_list
    df['V3PA_WFI_CEN [calc]'] = v3pa_wfi_list

    n_computed = sum(1 for v in ra_v1_list if v is not None)
    print(f"  📐 Computed V1 + WFI_CEN pointing for {n_computed}/{len(df)} rows")

    return df


import matplotlib
matplotlib.use('Agg')  # non-interactive backend for batch generation

def generate_sky_plot_pngs(opup_filepath, output_dir, df):
    """
    For each unique visit in the DataFrame, extract the .vst file from the OPUP,
    run roman_visit_viewer's plot_manager, and save a PNG.

    Returns:
        dict: mapping visit_filename (str) -> png_relative_path (str)
    """
    
    matplotlib.use('Agg')

    output_dir = Path(output_dir)
    png_dir = output_dir / "sky_plots"
    png_dir.mkdir(exist_ok=True)

    if 'Visit_File_Name' not in df.columns:
        print("Warning: No Visit_File_Name column; cannot generate sky plots.")
        return {}

    unique_vst = [v for v in df['Visit_File_Name'].dropna().unique() if str(v).endswith('.vst')]
    if not unique_vst:
        return {}

    # ── Use the existing function to bulk-extract all .vst contents at once ──
    visit_contents = get_all_visit_contents(opup_filepath, unique_vst)

    visit_png_map = {}

    for vst_name, vst_content in visit_contents.items():
        try:
            # 1) Write the visit content to a temp file so VisitFileParser can open it
            tmp_vst = png_dir / vst_name
            with open(tmp_vst, 'w') as f:
                f.write(vst_content)

            # 2) Parse and plot
            parser = VisitFileParser(str(tmp_vst))
            plot_manager(parser, exp_num=1, output_dir=png_dir)

            # 3) plot_manager saves to CWD as "<name>_all.png"
            #    Move it into our sky_plots directory
            vst_stem = vst_name.replace('.vst', '')
            for png_file in png_dir.glob(f"{vst_stem}*.png"):
                visit_png_map[vst_name] = str(png_file)
                print(f"  🔭 Generated sky plot: {png_file}")
                break

            # Clean up temp .vst
            tmp_vst.unlink(missing_ok=True)

        except Exception as e:
            print(f"  ⚠️  Could not generate sky plot for {vst_name}: {e}")
            # Clean up on failure too
            tmp_vst = png_dir / vst_name
            if tmp_vst.exists():
                tmp_vst.unlink(missing_ok=True)

    return visit_png_map

def generate_skyplot_mosaic_html(visit_png_map, opup_name, output_path, df=None):
    """
    Generate a standalone HTML mosaic page of all sky plot PNGs.
    Each image gets an anchor ID so the main report can deep-link to it.

    Args:
        visit_png_map: dict mapping visit_filename -> relative png path
        opup_name: str, OPUP name for the page title
        output_path: Path or str, where to write the HTML
        df: optional DataFrame to pull metadata (RA, Dec, filter) per visit

    Returns:
        Path to the generated HTML file, or None if no images
    """
    if not visit_png_map:
        return None

    output_path = Path(output_path)

    # Build metadata lookup from DataFrame if available
    visit_meta = {}
    if df is not None and 'Visit_File_Name' in df.columns:
        for vf in visit_png_map:
            rows = df[df['Visit_File_Name'] == vf]
            if len(rows) > 0:
                row = rows.iloc[0]
                meta = {}
                for col in ['Visit_ID', 'RA', 'DEC', 'Position_Angle',
                            'WFI_Optical_Element', 'Program_Number']:
                    if col in rows.columns and pd.notna(row.get(col)):
                        meta[col] = row[col]
                visit_meta[vf] = meta

    # Sort by visit filename for consistent ordering
    sorted_visits = sorted(visit_png_map.keys())

    n_visits = len(sorted_visits)

    # ── Build the HTML ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sky Plots - {opup_name}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a2e;
            color: #e0e0e0;
        }}
        h1 {{
            color: #e0e0e0;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 5px;
        }}
        .subtitle {{
            color: #95a5a6;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .back-link {{
            display: inline-block;
            margin-bottom: 15px;
            color: #3498db;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
        }}
        .back-link:hover {{
            color: #5dade2;
            text-decoration: underline;
        }}

        /* ── Filter bar ── */
        .filter-bar {{
            background: #16213e;
            padding: 12px 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .filter-bar label {{
            font-size: 13px;
            color: #95a5a6;
            font-weight: 600;
        }}
        .filter-bar input {{
            padding: 6px 12px;
            border: 1px solid #34495e;
            border-radius: 4px;
            background: #1a1a2e;
            color: #e0e0e0;
            font-size: 13px;
            width: 220px;
        }}
        .filter-bar input::placeholder {{
            color: #5a6a7a;
        }}
        .count-badge {{
            background: #3498db;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-left: auto;
        }}

        /* ── Mosaic grid ── */
        .mosaic {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
            gap: 16px;
        }}
        .card {{
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #2c3e6e;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            border-color: #3498db;
            box-shadow: 0 4px 20px rgba(52, 152, 219, 0.3);
        }}
        .card.highlight {{
            border-color: #f1c40f;
            box-shadow: 0 4px 24px rgba(241, 196, 15, 0.4);
        }}
        .card-header {{
            padding: 10px 14px;
            background: #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .card-title {{
            font-weight: 700;
            font-size: 13px;
            color: #e0e0e0;
            font-family: 'Consolas', 'Courier New', monospace;
        }}
        .card-meta {{
            font-size: 11px;
            color: #95a5a6;
        }}
        .card-meta span {{
            margin-left: 10px;
        }}
        .card img {{
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }}

        /* ── Lightbox ── */
        .lightbox {{
            display: none;
            position: fixed;
            z-index: 9999;
            left: 0; top: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.92);
            justify-content: center;
            align-items: center;
            cursor: zoom-out;
        }}
        .lightbox.active {{
            display: flex;
        }}
        .lightbox img {{
            max-width: 95%;
            max-height: 95%;
            border-radius: 6px;
            box-shadow: 0 0 40px rgba(0,0,0,0.8);
        }}
        .lightbox-title {{
            position: fixed;
            top: 15px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            font-size: 14px;
            font-weight: 600;
            background: rgba(0,0,0,0.6);
            padding: 6px 16px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>

    <a href="{opup_name}_report.html" class="back-link">← Back to OPUP Report</a>
    <h1>🔭 Visit Sky Plots</h1>
    <p class="subtitle">{opup_name} &mdash; {n_visits} visits</p>

    <div class="filter-bar">
        <label for="searchBox">Filter:</label>
        <input type="text" id="searchBox" placeholder="Type visit ID, filter, program..." oninput="filterCards()">
        <span class="count-badge" id="countBadge">{n_visits} of {n_visits}</span>
    </div>

    <div class="mosaic" id="mosaic">
"""

    # ── One card per visit ──
    for vst_name in sorted_visits:
        png_path = visit_png_map[vst_name]
        anchor_id = vst_name.replace('.vst', '')
        meta = visit_meta.get(vst_name, {})

        # Build metadata spans
        meta_spans = ""
        if 'Visit_ID' in meta:
            meta_spans += f'<span>ID: {meta["Visit_ID"]}</span>'
        if 'RA' in meta and 'DEC' in meta:
            meta_spans += f'<span>RA: {float(meta["RA"]):.4f}°</span>'
            meta_spans += f'<span>Dec: {float(meta["DEC"]):.4f}°</span>'
        if 'Position_Angle' in meta:
            meta_spans += f'<span>PA: {float(meta["Position_Angle"]):.2f}°</span>'
        if 'WFI_Optical_Element' in meta:
            meta_spans += f'<span>Filter: {meta["WFI_Optical_Element"]}</span>'
        if 'Program_Number' in meta:
            meta_spans += f'<span>Prog: {meta["Program_Number"]}</span>'

        # Searchable text blob (hidden, used by JS filter)
        search_blob = f'{vst_name} {anchor_id} ' + ' '.join(str(v) for v in meta.values())

        html += f"""        <div class="card" id="{anchor_id}" data-search="{search_blob.lower()}">
            <div class="card-header">
                <span class="card-title">{vst_name}</span>
                <span class="card-meta">{meta_spans}</span>
            </div>
            <img src="{png_path}" alt="Sky plot for {vst_name}" 
                 onclick="openLightbox(this.src, '{vst_name}')" loading="lazy">
        </div>
"""

    html += """    </div>

    <!-- Lightbox overlay for full-size viewing -->
    <div class="lightbox" id="lightbox" onclick="closeLightbox()">
        <span class="lightbox-title" id="lightboxTitle"></span>
        <img id="lightboxImg" src="" alt="Full size sky plot">
    </div>

    <script>
        function filterCards() {
            const query = document.getElementById('searchBox').value.toLowerCase();
            const cards = document.querySelectorAll('.card');
            let visible = 0;
            cards.forEach(function(card) {
                const text = card.getAttribute('data-search') || '';
                if (text.includes(query)) {
                    card.style.display = '';
                    visible++;
                } else {
                    card.style.display = 'none';
                }
            });
            document.getElementById('countBadge').textContent = visible + ' of ' + cards.length;
        }

        function openLightbox(src, title) {
            document.getElementById('lightboxImg').src = src;
            document.getElementById('lightboxTitle').textContent = title;
            document.getElementById('lightbox').classList.add('active');
        }

        function closeLightbox() {
            document.getElementById('lightbox').classList.remove('active');
        }

        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeLightbox();
        });

        // Highlight card if arrived via anchor
        (function() {
            var hash = window.location.hash.replace('#', '');
            if (hash) {
                var el = document.getElementById(hash);
                if (el) {
                    el.classList.add('highlight');
                    setTimeout(function() { el.scrollIntoView({behavior: 'smooth', block: 'center'}); }, 100);
                    setTimeout(function() { el.classList.remove('highlight'); }, 3000);
                }
            }
        })();
    </script>

</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  🖼️  Generated sky plot mosaic: {output_path}")
    return output_path

def parse_visit_header(visit_header_line):

    visit_info = {}
    items = [x.strip() for x in visit_header_line.split(',')]

    visit_info['Visit_ID'] = items[1].replace('V','') if items[1].startswith('V') and items[1][1:].isdigit() else None

    for item in items:
        if '=' in item:
            key, val = item.split('=')
            visit_info[key] = val
    
    return visit_info


def extract_exposure_metadata(parsed_visit):
    '''
    This function takes the parsed visit file output and interprets the metadata
    corresponding to each science exposure.
    
    PARAMS:
    -------
    parsed_visit - information extracted from the visit file using the parse_visit_file function

    RETURNS:
    -------
    Pandas dataframe with metadata info for each science exposure.

    '''
    
    # Initialize empty dict to store snapshot of current metadata:
    current_metadata = {}

    # Initialize visit info for the given visit
    visit_info = parsed_visit.pop('visit')

    # Initalize list of exposure metadata
    exposures = []

    # Iterate through each group in the visit
    for group_num, group in parsed_visit.items():
        if isinstance(group, dict):
            # Iterate each sequence within the group
            for seq_num in group['sequences']:
                sequence = group[seq_num]

                # Interate each activity within the sequence
                for act_num in sequence['activities']:
                    activity = sequence[act_num]
                    
                    # Grab the parameters from the current activity's command
                    # params = parsed_visit[group_num][seq_num][act_num]['command']
                    params = activity['command']
        
                    # Update current metadata parameters
                    current_metadata.update(params)
                    
                    # Check if this activity is the start of an exposure
                    if params['command'] in ['WFI_EXPOSURE_START_F', 'WFI_SET_USER_ID_SCI_EXPOSE_F']:
        
                        # Save group, seq, and act numbers to current metadata
                        current_metadata.update({
                                                'GROUP': group_num,
                                                'SEQUENCE': seq_num,
                                                'ACTIVITY': act_num})

                        # Add visit ID, early start, late start, and late end times
                        current_metadata.update(visit_info)

                        # Save target type if it exists:
                        if 'TARGET_TYPE' in parsed_visit.keys():    
                            current_metadata.update({'TARGET_TYPE': parsed_visit['TARGET_TYPE']})
        
                        # Save all current metadata parameters in exposure list
                        # as a copy of the dict to avoid overwriting data.
                        exposures.append(current_metadata.copy())


    # Convert list of exposures to pandas dataframe
    exp_df = pd.DataFrame(exposures)
    
    # # Setting the science ID to be the dataframe index
    # if 'SCI_ID' in exp_df.columns:
    #     exp_df.set_index('SCI_ID', inplace=True)
    # elif 'SCI_USER_ID' in exp_df.columns:
    #     exp_df.set_index('SCI_USER_ID', inplace=True)
    
    return exp_df

def parse_visit_file_obj(file_obj):
    '''
    This function parses a visit file object and returns a nested dictionary
    containing metadata info for the groups, sequences, and activities
    defined in the visit.

    PARAMS:
    -------
    file_obj - file object of the visit file


    RETURNS:
    -------
    visit - dict, nested dictionary containing metadata info. Groups,
            sequences, and activities are separated by their numbers.
            e.g. visit[1] would return info for the first group, visit[1][1]
            would return info for the first sequence of the first group, and
            visit[1][1][1] would return info for the first activity of the
            first sequence of the first group, all in the form of a dictionary.
    '''
    
    # Initializing dict to store visit info
    visit = {}

    # Iterate by line in the visit file
    for line in file_obj:
        
        
        # Remove trailing white space
        line = line.strip()

        if 'Target Type' in line:
            target_type_match = re.match(r'.* Target Type = (.+)', line)
            if target_type_match:
                visit['TARGET_TYPE']=target_type_match.group(1)
        
        # Parse VISIT line
        if line.startswith('VISIT'):
            visit['visit'] = parse_visit_header(line)

            # Initialize list of group number as empty list
            visit['groups'] = []

        ### Parse GROUP line
        elif line.startswith('GROUP'):
            group_match = re.match(r'GROUP, (\d+), CONGRP=(.*);', line)
            if group_match:
                # Extracting the group number
                group_num = int(group_match.group(1))

                # Creating dict for the extracted group info
                current_group = {'group_num': group_num, 'congrp': group_match.group(2), 'sequences': []}

                # Adding group number to list of groups
                visit['groups'].append(group_num)

                # Adding current group to visit
                visit[group_num] = current_group

        ### Parse SEQ line
        elif line.startswith('SEQ'):
            seq_match = re.match(r'SEQ, (\d+), CONSEQ=(.*);', line)
            if seq_match:
                # Extracting the sequence number
                seq_num = int(seq_match.group(1))

                # Creating dict for the extracted sequence info
                current_seq = {'seq_num': seq_num, 'conseq': seq_match.group(2), 'activities': []}

                # Adding sequence number to list of sequences
                visit[group_num]['sequences'].append(seq_num)


                # Adding current sequence to group
                visit[group_num][seq_num] = current_seq

        ### Parse ACT line
        elif line.startswith('ACT'):
            # Regex match for the activity definition
            act_match = re.match(r'ACT,\s+(\d+),(.*)', line)
            if act_match:
                # Extracting the activity number
                act_num = int(act_match.group(1))

                # Creating dict for the extracted activity info
                current_act = {
                    'act_num': act_num,
                    'command': parse_visit_command(act_match.group(2))
                }

                # Adding new activity number to activity list:
                visit[group_num][seq_num]['activities'].append(act_num)

                # Adding activity 
                visit[group_num][seq_num][act_num] = current_act


    
    # returning visit information
    return visit


def parse_visit_file(filepath):
    '''
    This function parses a visit file and returns a nested dictionary
    containing metadata info for the groups, sequences, and activities
    defined in the visit.

    PARAMS:
    -------
    filepath - str, path to the visit file


    RETURNS:
    -------
    visit - dict, nested dictionary containing metadata info. Groups,
            sequences, and activities are separated by their numbers.
            e.g. visit[1] would return info for the first group, visit[1][1]
            would return info for the first sequence of the first group, and
            visit[1][1][1] would return info for the first activity of the
            first sequence of the first group, all in the form of a dictionary.
    '''
    
    try:
        # Open the visit file for parsing:
        visit_file_obj = read_visit_file(filepath)

        # Parsing the visit file object
        visit = parse_visit_file_obj(visit_file_obj)
        
        # Converting to data frame
        visit_df = extract_exposure_metadata(visit)

        # returning visit information
        return visit_df
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        print(f'Encountered error while processing visit {filepath}: {e}')
        return pd.DataFrame()



def parse_visit_command(command_str):
    '''
    This function parses a given activity command and returns a dictionary of command
    parameters.

    PARAMS:
    command_str - str, command as it is written in the visit file


    RETURNS:
    command - dict, dictionary of metadata info for the given command
    '''

    try:
        
        # Remove trailing white space in command
        command_str = command_str.strip()
    
        # Initialize empty dictionary with command info
        command = {}
    
        if ';' in command_str:
            # Splitting command by semicolon
            command_str_split = [x for x in command_str.split(';') if len(x)>0]
            
            if len(command_str_split)==2:
                command_function = command_str_split[0]
                command_comments = command_str_split[1]
            else:
                # If there are multiple semicolons on the line, it means that
                # a command has likely been temporarily commented out and 
                # replace with a wait, in which case we only save the function.
                command_function = command_str_split[0]
                command_comments = ''
            
            
        else:
            command_function = command_str
            command_comments = ''
    
        # Getting the parameter names if available:
        cmd_param_names = command_comments.strip().split(',')
    
        # Regex match for the activity definition
        cmd_match = re.match(r'(\w+)\((.+)\)', command_function.strip())

        ### Extracting command info if the regex could be matched.
        if cmd_match:
            # Extracting command info
            cmd_name = cmd_match.group(1)
            cmd_params = cmd_match.group(2).strip().split(',')
    
            # Adding command name to the dict
            command['command']=cmd_name
    
            # There are parameters passed in the function call
            params_present=True
            
            # If the parameter names could be found
            if (cmd_param_names[0]!='') and (len(cmd_params)==len(cmd_param_names)):
                param_names_present=True
            else:
                param_names_present=False
    
        elif '(' not in command_str:
            # If no parameters are passed in the command, only return the command name
            command['command'] = command_str.replace(';', '')
    
            # No parameters are present
            params_present=False
            param_names_present=False
    
        else:
            # If the format could not be recognized, return the command as written
            command['command'] = command_str
    
            # Assume no parameters to parse
            params_present=False
            param_names_present=False
    
    
        # Adding the parameter names and parameters to the dict if available.
        # (If either are unavailable, do nothing).
        if params_present and param_names_present:
            # If the command has parameters and the names of the 
            # parameters are included after the semicolon, we can
            # use the parameter names in the metadata.
    
            # Checking that the parameter names and parameters match
            if len(cmd_params)==len(cmd_param_names):
                # Parsing command parameters and parameter names
                for key, val in zip(cmd_param_names, cmd_params):
                    
                    try:
                        command[key.strip()] = literal_eval(val.strip())
                    except ValueError:
                        command[key.strip()] = val.strip()
                    except SyntaxError:
                        command[key.strip()] = val.strip()
                        
                # This dict contains information for special commands that are usually executed multiple
                # times prior to a science exposure. Since the parameters will be overwritten for each
                # consecutive run, we need to rename the parameters to be unique.
                specials = {
                            'WFI_SRCS':{'key':'BANK', 'desc':'BANK'},
                            'FGS_GSDS_ENTRY':{'key':'WFI_DET', 'desc':'GW'},
                            'SCE_GW_CONFIG_LOC':{'key':'SCENUM','desc':'GW'}
                            }

                # Checking to see if the command is one of the special commands requiring the parameters to be
                # renamed to avoid being overwritten.
                for spec_cmd in specials:
                    if spec_cmd in cmd_name:
                        command = rename_command_params(command, specials[spec_cmd]['key'], specials[spec_cmd]['desc'])
                        command['command'] = command_str

        elif params_present:
            # If the command has parameters, but the names of the parameters
            # could not be found, only pass the parameters.
    
            for key, val in zip([f'PARAM{n:003}' for n in range(1,len(cmd_params))], cmd_params):
                try:
                    command[key] = literal_eval(val)
                except ValueError:
                    command[key] = val

        
        # Returning command dict
        return command
    except:
        print(f'Could not parse command: {command_str}')
        import traceback
        traceback.print_exc()
        return None

def rename_command_params(command, key, desc):
    to_add = command[key]
    command = {f'{x}_{desc}{to_add:02}':y for x,y in command.items() if x not in [key, 'command']}
    return command
    

def get_SCF_from_OPUP(opup_filepath):

    with tarfile.open(opup_filepath, 'r:gz') as tarball_file:
        
        # Search the tarball members for the given visit ID:
        scf_files = [os.path.join(opup_filepath, x) for x in tarball_file.getnames() if x.startswith('SCF')]
    
    return scf_files

def get_manifest_from_OPUP(opup_filepath):

    with tarfile.open(opup_filepath, 'r:gz') as tarball_file:
        
        # Search the tarball members for the given visit ID:
        manifest_filepaths = [os.path.join(opup_filepath, x) for x in tarball_file.getnames() if x.endswith('.man')]

    return manifest_filepaths

def get_odf_from_OPUP(opup_filepath):

    with tarfile.open(opup_filepath, 'r:gz') as tarball_file:
        
        # Search the tarball members for the given visit ID:
        odf_filepaths = [os.path.join(opup_filepath, x) for x in tarball_file.getnames() if 'odf' in x and x.endswith('.json')]

    return odf_filepaths

def get_visits_from_SCF(scf_filepath):
    
    # If the SCF is within a gzipped OPUP
    if not Path(scf_filepath).parent.is_dir() and tarfile.is_tarfile(Path(scf_filepath).parent.as_posix()):
        with tarfile.open(os.path.dirname(scf_filepath), 'r:gz') as tarball_file:
            # Extracting file as object
            scf_file = tarball_file.extractfile(os.path.basename(scf_filepath))

            # Accessing the gzipped scf file object within the gzipped opup
            with tarfile.open(fileobj=scf_file, mode='r:gz') as tarball_file:
                
                # Search the tarball members for visit files:
                visit_files = [os.path.join(scf_filepath, x) for x in tarball_file.getnames() if x.endswith('.vst')]
    elif tarfile.is_tarfile(scf_filepath):
        # Accessing the gzipped scf file object
        with tarfile.open(scf_filepath, mode='r:gz') as tarball_file:
            
            # Search the tarball members for visit files:
            visit_files = [os.path.join(scf_filepath, x) for x in tarball_file.getnames() if x.endswith('.vst')]
    else:
        visit_files = [os.path.join(scf_filepath, x) for x in os.listdir(scf_filepath) if x.endswith('.vst')]
    
    return visit_files

def get_ops_from_SCF(scf_filepath):

    with tarfile.open(scf_filepath, 'r:gz') as tarball_file:
        
        # Search the tarball members for the given visit ID:
        ops_files = [os.path.join(scf_filepath, x) for x in tarball_file.getnames() if x.endswith('.ops')]

    return ops_files

def parse_opup_manifest(opup_filepath):

    # Getting the filepath of the manifest file
    manifest_filepaths = get_manifest_from_OPUP(opup_filepath)
    
    # Parsing each manifest and returning the info as a list of files
    manifest_info = []
    for manifest_filepath in manifest_filepaths:
        manifest_info.extend(parse_manifest(manifest_filepath))
    
    return manifest_info

def parse_manifest(manifest_filepath):

    # Extracting manifest output
    with tarfile.open(os.path.dirname(manifest_filepath), 'r:gz') as tarball_folder:
        f = tarball_folder.extractfile(os.path.basename(manifest_filepath))

        # Reading reach line and decoding
        manifest_lines = [x.decode() for x in f]
        manifest_output = [line.strip() for line in manifest_lines if line and not line.startswith('#')]

    return manifest_output


def read_visit_file(visit_filepath):
    """This function accepts a visit file path and returns a StringIO object containing the visit file text.
    The file path can be within gzipped SCF and OPUP archives, or not.

    Args:
        visit_filepath (str): Path to the visit file.

    Returns:
        StringIO: StringIO stream containing the visit file text.
    """

    
    ### Identifying if the visit file path is contained in an SCF and OPUP directory
    if Path(visit_filepath).parent.stem.startswith('SCF'):
        scf_filepath = Path(visit_filepath).parent.as_posix()
    else:
        scf_filepath = ''

    if Path(scf_filepath).parent.stem.endswith('opup'):
        # opup_filepath = os.path.dirname(scf_filepath)
        opup_filepath = Path(scf_filepath).parent.as_posix()
    else:
        opup_filepath = ''


    ### Identifying if those directories are gzipped

    # Determine if the opup is gzipped
    if opup_filepath !='':
        opup_gzipped = tarfile.is_tarfile(opup_filepath)
    else:
        opup_gzipped = False
    
    if opup_gzipped:
        # Opening the opup tarball
        opup_tarball = tarfile.open(opup_filepath, 'r:gz')
        # opup_tarball = io.BytesIO(tarfile.open(opup_filepath, 'r:gz'))

        # Extracting SCF file
        scf_file = opup_tarball.extractfile(Path(scf_filepath).name)
        # scf_file = io.BytesIO(opup_tarball.extractfile(Path(scf_filepath).name).read())

        # Determine if the scf file object is gzipped too
        scf_gzipped = tarfile.is_tarfile(scf_file)
        scf_file.seek(0)
    else:
        if scf_filepath !='':
            # Determing if the scf file path is gzipped
            scf_gzipped = tarfile.is_tarfile(scf_filepath)
        else:
            scf_gzipped = False

    # print(visit_filepath)
    # print(opup_filepath)
    # print(scf_filepath)
    # print(opup_gzipped, scf_gzipped, tarfile.is_tarfile(scf_file))

    ### Extracting the info from the visit file and creating a StringIO object containing it.

    if scf_gzipped:
        if opup_gzipped:
            # This covers the case when both the OPUP and SCF are gzipped.
            # Accessing the gzipped scf file object within the gzipped opup
            with tarfile.open(fileobj=scf_file, mode='r:gz') as scf_tarball:
                # Extracting visit file
                visit_file_ex = scf_tarball.extractfile(Path(visit_filepath).name)

                # Getting visit file text
                visit_file_obj= io.StringIO(visit_file_ex.read().decode())

            # Closing scf tarball
            scf_file.close()

            # Closing opup tarball
            opup_tarball.close()

        else:
            # This covers the case when just the SCF is gzipped
            # If there is no gzipped OPUP folder, then we can access the gzipped SCF directly.
            with tarfile.open(scf_filepath, mode='r:gz') as scf_tarball:
                # Extracting visit file
                visit_file_ex = scf_tarball.extractfile(Path(visit_filepath).name)

                # Getting visit file text
                visit_file_obj= io.StringIO(visit_file_ex.read().decode())
    else:
        # This covers the case if the visit file is not in a gzipped archive at all:
        with open(visit_filepath, 'r') as visit_file_ex:
            visit_file_obj = io.StringIO(visit_file_ex.read())

    

    return visit_file_obj

def process_odf_files(json_files):
    """
    Process a list of JSON files and extract visit data.

    Args:
    json_files (list): A list of file paths to JSON files.

    Returns:
    list: A list of pandas DataFrames, each containing visit data from a JSON file.
    """
    outputs = []
    for json_file in json_files:
        
        opup_filepath = Path(json_file).parent.as_posix()
        if tarfile.is_tarfile(opup_filepath):
            # If the json file is in a gzipped tarball, we need to extract the file object
            with tarfile.open(opup_filepath, mode='r:gz') as opup_tarball:
                # Extracting json file
                json_file_ex = opup_tarball.extractfile(Path(json_file).name)

                # Getting json file text as a string IO
                json_file= io.StringIO(json_file_ex.read().decode())

        # Read the JSON file
        data = pd.read_json(json_file)

        # Extract the 'visits' data
        visits = data['visits']

        # Convert each visit to a pandas Series and create a DataFrame
        dataf = pd.DataFrame([pd.Series(visit) for visit in visits])

        # Add a column with the name of the parent directory (assumed to be 'opup')
        dataf['opup'] = os.path.basename(opup_filepath)

        # Append the DataFrame to the outputs list
    outputs.append(dataf)

    return outputs


def export_obsplan_from_json(json_files):
    """
    Export observation plan data from JSON files to an Excel file.

    Args:
    json_files (list): A list of file paths to JSON files containing observation plan data.

    Returns:
    pd.DataFrame: A DataFrame containing the combined observation plan data from all input JSON files.
    """
    outputs = process_odf_files(json_files)
    output = pd.concat(outputs, ignore_index=True)
    return output

def parse_OPUP(opup_filepath):

    # Getting a list of all visit file paths in the opup
    scf_files = get_SCF_from_OPUP(opup_filepath)

    # Parsing each SCF
    opup_info = pd.DataFrame()
    for scf_filepath in scf_files:
        scf_info = parse_SCF(scf_filepath)

        opup_info = pd.concat((opup_info, scf_info))

    # Parsing the obsplan json
    obsplan_filepaths = get_odf_from_OPUP(opup_filepath)
    obsplan_df= export_obsplan_from_json(obsplan_filepaths)

    # Merging the obsplan df with the opup df
    opup_info = pd.merge(obsplan_df, opup_info, on=['Visit_ID'])

    return opup_info
#%%
def parse_SCF(scf_filepath):
    try:
        
        # Get list of visit file paths from the given SCF
        visit_files = get_visits_from_SCF(scf_filepath)

        # Parsing each visit file and adding to a pandas df
        scf_info = pd.DataFrame()
        for visit_file in visit_files:

            # Parsing visit file
            visit_df = parse_visit_file(visit_file)

            # Concatenating df to opup info df
            scf_info = pd.concat((scf_info, visit_df))
        
    except Exception as e:
        print(f'Encountered error while processing {scf_filepath}: {e}')
        scf_info = pd.DataFrame()

    return scf_info


def get_current_gw_columns(df):
    """Returns a list of Guide Window related columns in the given data frame. This is useful for separating the GW columns.


    Args:
        df (pd.DataFrame): DataFrame containing the Guide Window columns to identify.

    Returns:
        list: List of guide window column names in the given DataFrame.
    """

    gw_columns=['ACQ_USE',
                'ACQ_H',
                'ACQ_V',
                'TRK_USE',
                'TRK_H',
                'TRK_V',
                'EDGE',
                'LOW',
                'NOM',
                'HIGH',
                'SKYBGND',
                'STAR_TEMP',
                'X_START',
                'Y_START']
    
    current_gw_cols = []
    
    for x in range(1,19):
        for col in gw_columns:
            test_col = f'{col}_GW{x:02}'
            if test_col in df.columns:
                    current_gw_cols.append(f'{col}_GW{x:02}')

    return current_gw_cols

def split_df_columns(df, columns_to_split):

    # Creating new DataFrame with the split columns
    df_2 = df[columns_to_split]

    # Dropping split columns from original DataFrame
    df_1 = df.drop(columns=columns_to_split)
    
    return df_1, df_2

def prioritize_columns(df, priority_columns):
    # Only keep priority columns that actually exist in the DataFrame
    existing_priority = [col for col in priority_columns if col in df.columns]
    new_order = existing_priority + [col for col in df.columns if col not in existing_priority]
    return df[new_order]

def find_nontgz_parent(folderpath):
    # Returning the first path in the parents of the given path
    # that is a directory.
    for path in Path(folderpath).parents:
        if Path(path).is_dir():
            return path

def write_to_CSV(df, output_csv, keep_GW=True):
    '''
    This function writes a given pandas DataFrame to a CSV file.
    If the dataframe has Guide Window information, this function will
    extract this information to a separate CSV.

    PARAMS:
    df - pandas DataFrame
    output_csv - str, path to the output CSV file
    keep_GW - bool, whether to keep Guide Window columns in the main CSV

    RETURNS:
    None
    '''
    # Convert Path to string if necessary
    output_csv = str(output_csv)
    
    # Getting guide window columns
    gw_cols = get_current_gw_columns(df)

    if len(gw_cols)>0:
        # Splitting the Guide Window info from the main dataframe
        df_out, gw_df  = split_df_columns(df, gw_cols)

        # Writing Guide Window info to separate CSV
        gw_df.to_csv(output_csv.replace('.csv', '_GWInfo.csv'), index=False)

        if keep_GW:
            # Writing to CSV with GW info
            df.to_csv(output_csv, index=False)
        else:
            # Writing to CSV without GW info
            df_out.to_csv(output_csv, index=False)
    else:
        # Writing to CSV
        df.to_csv(output_csv, index=False)


def process_OPUPs(opup_filepaths, output_dir=None, keep_GW=True):

    # Iterating through list of opup filepaths
    for opup_filepath in opup_filepaths:
        # If the output directory is not given, then we use the local directory.
        # If the local directory is a file (i.e. gzipped archive), then we use the closest non-gzipped parent.
        if output_dir is None:
            output_dir = Path(opup_filepath).parent.as_posix()

            # If the SCF file is stored in a gzipped archive, we have to modify the save path to accommodate this.
            if not Path(output_dir).is_dir():
                output_dir = find_nontgz_parent(output_dir)

        output_csv = Path(output_dir).joinpath(Path(opup_filepath).stem + '_csv.csv').as_posix()

        # Parse the opup to get the opup info
        opup_info = parse_OPUP(opup_filepath)

        # Move priority columns to the left-side of the data frame
        opup_info = prioritize_columns(opup_info, PRIORITY_COLUMNS)

        # Write to CSV
        if len(opup_info.columns)>0:
            write_to_CSV(opup_info, output_csv, keep_GW=keep_GW)
        else:
            print(f'No columns were returned for {opup_filepath}')

def process_SCFs(scf_filepaths, output_dir=None, keep_GW=True):

    # Iterating through list of SCF file paths
    for scf_filepath in scf_filepaths:
        # If the output directory is not given, then we use the local directory.
        # If the local directory is a file (i.e. gzipped archive), then we use the closest non-gzipped parent.
        if output_dir is None:
            output_dir = Path(scf_filepath).parent.as_posix()

            # If the SCF file is stored in a gzipped archive, we have to modify the save path to accommodate this.
            if not Path(output_dir).is_dir():
                output_dir = find_nontgz_parent(output_dir)

        output_csv = Path(output_dir).joinpath(Path(scf_filepath).stem + '_csv.csv').as_posix()

        # Parsing SCF
        scf_info = parse_SCF(scf_filepath)

        # Move priority columns to the left-side of the data frame
        scf_info = prioritize_columns(scf_info, PRIORITY_COLUMNS)

        # Writing to CSV
        if len(scf_info.columns)>0:
            write_to_CSV(scf_info, output_csv, keep_GW=keep_GW)
        else:
            print(f'No columns were returned for {scf_filepath}')

def process_visits(visit_filepaths, output_dir=None, keep_GW=True):

    # Iterating by visit file path
    for visit_filepath in visit_filepaths:

        # If the output directory is not given, then we use the local directory.
        # If the local directory is gzipped, we move to the closest non-gzipped parent.
        if output_dir is None:
            output_dir = Path(visit_filepath).parent.as_posix()

            # If the visit file is stored in a gzipped archive, we have to modify the save path to accommodate this.
            if not Path(output_dir).is_dir():
                output_dir = find_nontgz_parent(output_dir)

        # Creating output csv file path
        output_csv = Path(output_dir).joinpath(Path(visit_filepath).stem + '_csv.csv').as_posix()

        # Parsing visit file
        visit_info = parse_visit_file(visit_filepath)

        # Move priority columns to the left-side of the data frame
        visit_info = prioritize_columns(visit_info, PRIORITY_COLUMNS)

        if len(visit_info.columns) > 0:
            write_to_CSV(visit_info, output_csv, keep_GW=keep_GW)
        else:
            print(f'No columns were returned for {visit_filepath}')

#%%

def export_opup_to_html(opup_filepath, output_html_path=None, keep_GW=True):
    """
    Export OPUP data to an HTML file with hyperlinks to individual visit files.
    
    Args:
        opup_filepath: Path to the OPUP .tgz archive
        output_html_path: Path for output HTML file (optional)
        keep_GW: Whether to keep Guide Window columns
    
    Returns:
        Path to generated HTML file
    """
    # Parse the OPUP (reuse your existing function)
    opup_info = parse_OPUP(opup_filepath)
    
    # Prioritize columns
    opup_info = prioritize_columns(opup_info, PRIORITY_COLUMNS)
    
    # Remove GW columns if requested
    if not keep_GW:
        gw_cols = get_current_gw_columns(opup_info)
        opup_info = opup_info.drop(columns=gw_cols)
    
    # Determine output path
    if output_html_path is None:
        output_dir = Path(opup_filepath).parent
        if not output_dir.is_dir():
            output_dir = find_nontgz_parent(opup_filepath)
        output_html_path = Path(output_dir) / f"{Path(opup_filepath).stem}_report.html"
    
    # Generate HTML
    html_content = generate_html_report(opup_info, opup_filepath)
    
    # Write HTML file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_html_path}")
    return output_html_path

def get_all_visit_contents(opup_filepath, visit_filenames):
    """
    Extract contents of all visit files from the OPUP archive.
    Visit files are nested inside SCF tarballs within the OPUP.
    
    Args:
        opup_filepath: Path to the OPUP .tgz archive
        visit_filenames: List of visit filenames to extract
    
    Returns:
        Dictionary mapping visit_filename -> content
    """
    visit_contents = {}
    
    try:
        if tarfile.is_tarfile(opup_filepath):
            with tarfile.open(opup_filepath, 'r:gz') as opup_tar:
                # Find all SCF tarballs in the OPUP
                scf_members = [m for m in opup_tar.getmembers() if 'SCF' in m.name and m.name.endswith('.tgz')]
                
                print(f"Found {len(scf_members)} SCF tarball(s) in OPUP")
                
                # Process each SCF tarball
                for scf_member in scf_members:
                    print(f"  Processing {scf_member.name}...")
                    
                    # Extract the SCF tarball file object
                    scf_file_obj = opup_tar.extractfile(scf_member)
                    
                    if scf_file_obj:
                        # Open the nested SCF tarball
                        with tarfile.open(fileobj=scf_file_obj, mode='r:gz') as scf_tar:
                            # Look for visit files in this SCF
                            for member in scf_tar.getmembers():
                                filename = Path(member.name).name
                                
                                if filename in visit_filenames and filename not in visit_contents:
                                    # Extract the visit file content
                                    visit_file_obj = scf_tar.extractfile(member)
                                    if visit_file_obj:
                                        content = visit_file_obj.read().decode('utf-8', errors='replace')
                                        visit_contents[filename] = content
                                        print(f"    Extracted {filename} ({len(content)} chars)")
                
                # Mark any missing files
                for visit_filename in visit_filenames:
                    if visit_filename not in visit_contents:
                        visit_contents[visit_filename] = "Visit file not found in archive"
                        print(f"  Warning: {visit_filename} not found")
                        
    except Exception as e:
        print(f"Error reading archive {opup_filepath}: {e}")
        import traceback
        traceback.print_exc()
    
    return visit_contents

def extract_visit_file_contents(opup_filepath, visit_filename):
    """
    Extract the content of a specific visit file from the OPUP archive.
    Visit files are nested inside SCF tarballs within the OPUP.
    
    Args:
        opup_filepath: Path to the OPUP .tgz archive
        visit_filename: Name of the visit file to extract (e.g., 'V0010801001001006002.vst')
    
    Returns:
        String content of the visit file, or None if not found
    """
    try:
        if tarfile.is_tarfile(opup_filepath):
            with tarfile.open(opup_filepath, 'r:gz') as opup_tar:
                # Find all SCF tarballs in the OPUP
                scf_members = [m for m in opup_tar.getmembers() if 'SCF' in m.name and m.name.endswith('.tgz')]
                
                # Search each SCF tarball for the visit file
                for scf_member in scf_members:
                    scf_file_obj = opup_tar.extractfile(scf_member)
                    
                    if scf_file_obj:
                        # Open the nested SCF tarball
                        with tarfile.open(fileobj=scf_file_obj, mode='r:gz') as scf_tar:
                            # Look for the visit file
                            for member in scf_tar.getmembers():
                                if member.name.endswith(visit_filename):
                                    visit_file_obj = scf_tar.extractfile(member)
                                    if visit_file_obj:
                                        content = visit_file_obj.read().decode('utf-8', errors='replace')
                                        return content
        
        return None
        
    except Exception as e:
        print(f"Error extracting {visit_filename}: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_aladin_css():
    """Returns CSS for the Aladin Lite panel. Plain string, not f-string."""
    return """
        /* ── Aladin Lite Panel ── */
        #aladin-panel {
            position: fixed;
            bottom: 0;
            right: 0;
            width: 560px;
            height: 520px;
            background: #1a1a2e;
            border: 2px solid #3498db;
            border-radius: 12px 0 0 0;
            z-index: 2000;
            display: none;
            flex-direction: column;
            box-shadow: 0 -4px 30px rgba(0, 0, 0, 0.6);
            transition: width 0.3s ease, height 0.3s ease;
            resize: both;
            overflow: hidden;
        }
        #aladin-panel.expanded {
            width: 800px;
            height: 700px;
        }
        #aladin-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 14px;
            background: #0f3460;
            border-radius: 12px 0 0 0;
            cursor: grab;
            flex-shrink: 0;
            user-select: none;
        }
        #aladin-panel-header:active {
            cursor: grabbing;
        }
        #aladin-panel-header h3 {
            margin: 0;
            font-size: 13px;
            color: #e0e0e0;
            font-family: 'Consolas', 'Courier New', monospace;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .aladin-panel-controls {
            display: flex;
            gap: 4px;
            flex-shrink: 0;
        }
        .aladin-panel-controls button {
            background: none;
            border: 1px solid #3498db;
            color: #3498db;
            border-radius: 4px;
            padding: 2px 8px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
            line-height: 1.4;
        }
        .aladin-panel-controls button:hover {
            background: #3498db;
            color: white;
        }
        #aladin-container {
            flex: 1;
            min-height: 0;
            position: relative;
        }
        #aladin-info-bar {
            padding: 5px 14px;
            background: #0d1b36;
            color: #95a5a6;
            font-size: 11px;
            font-family: 'Consolas', 'Courier New', monospace;
            display: flex;
            justify-content: space-between;
            flex-shrink: 0;
            flex-wrap: wrap;
            gap: 4px;
            border-top: 1px solid #2c3e6e;
        }
        #aladin-info-bar .coord-val {
            color: #3498db;
            font-weight: bold;
        }
        #aladin-survey-label {
            color: #f1c40f;
            font-weight: bold;
        }
        tr[data-fp-key] {
            cursor: pointer;
        }
        tr[data-fp-key]:hover td {
            background-color: rgba(52, 152, 219, 0.08) !important;
        }
        tr.aladin-active {
            outline: 2px solid #f1c40f;
            outline-offset: -2px;
        }
        tr.aladin-active td {
            background-color: rgba(241, 196, 15, 0.06) !important;
        }
        #aladin-open-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #0f3460;
            color: #3498db;
            border: 2px solid #3498db;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            font-size: 22px;
            cursor: pointer;
            z-index: 1999;
            display: none;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            transition: all 0.2s;
        }
        #aladin-open-btn:hover {
            background: #3498db;
            color: white;
            transform: scale(1.1);
        }
"""

def get_aladin_html():
    """Returns the HTML for the Aladin panel. Plain string."""
    return """
    <button id="aladin-open-btn" onclick="openAladinPanel()" title="Open WFI Sky Viewer">&#x1F52D;</button>
    <div id="aladin-panel">
        <div id="aladin-panel-header">
            <h3>&#x1F52D; <span id="aladin-title">WFI Sky Viewer</span></h3>
            <div class="aladin-panel-controls">
                <button onclick="selectAllFootprints()" title="Select all visits">All</button>
                <button onclick="clearAllFootprints()" title="Clear all footprints">&#x1F5D1;</button>
                <button onclick="toggleAladinSurvey()" title="Cycle survey background">&#x1F5FA;</button>
                <button onclick="toggleSCALabels()" title="Toggle SCA labels" id="label-toggle-btn">&#x1F3F7;</button>
                <button onclick="toggleGuideStars()" title="Toggle guide stars" id="gs-toggle-btn">&#x2B50;</button>
                <button onclick="toggleAladinExpand()" title="Expand / Shrink" id="expand-btn">&#x29e2;</button>
                <button onclick="closeAladinPanel()" title="Close panel">&#x2715;</button>
            </div>
        </div>
        <div id="aladin-container"></div>
        <div id="aladin-info-bar">
            <span>Visit <span id="aladin-visit-id" class="coord-val">&mdash;</span></span>
            <span>RA <span id="aladin-ra" class="coord-val">&mdash;</span>&deg;</span>
            <span>Dec <span id="aladin-dec" class="coord-val">&mdash;</span>&deg;</span>
            <span>PA <span id="aladin-pa" class="coord-val">&mdash;</span>&deg;</span>
            <span id="aladin-survey-label">DSS2</span>
        </div>
    </div>
"""

def add_aladin_data_attributes_to_row(tr_tag, row):
    """
    Splice data-q1..q4, data-ra, data-dec, data-pa, data-visit-id,
    and data-sci-id into an opening <tr ...> tag.

    The JavaScript will use the quaternion to look up precomputed
    footprints keyed by 'VisitID_SciID'.

    Args:
        tr_tag: str  — e.g. '<tr class="group-parent-row">'
        row:    pandas Series with QUAT_COLS and/or RA, DEC, Position_Angle
    Returns:
        str  — modified <tr> tag
    """
    try:
        vid = str(row.get('Visit_ID', '')).strip()
        sid = str(row.get('SCI_ID', '')).strip()
        if not vid:
            return tr_tag

        # Build the same key used in precompute_wfi_footprints()
        row_key = f"{vid}_{sid}" if sid else vid

        parts = [f'data-fp-key="{row_key}"']
        parts.append(f'data-visit-id="{vid}"')
        if sid:
            parts.append(f'data-sci-id="{sid}"')

        # Quaternion (preferred)
        have_q = True
        for qc in QUAT_COLS:
            val = row.get(qc, None)
            try:
                fval = float(val)
                if np.isnan(fval):
                    have_q = False
                    break
                short = qc.replace('TAR_', '').replace('_ECI2BCS', '').lower()
                parts.append(f'data-{short}="{fval}"')
            except (ValueError, TypeError):
                have_q = False
                break

        # Always include RA/Dec/PA for info-bar display (even if derived)
        for col, attr in [('RA', 'ra'), ('DEC', 'dec'), ('Position_Angle', 'pa')]:
            val = row.get(col, None)
            try:
                parts.append(f'data-{attr}="{float(val)}"')
            except (ValueError, TypeError):
                pass

        attrs = ' '.join(parts)
        return tr_tag.replace('>', ' ' + attrs + '>', 1)

    except Exception:
        return tr_tag
    

def get_aladin_javascript(wfi_footprints_json):
    return f"""
    <link rel="stylesheet"
          href="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.min.css" />
    <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"
            charset="utf-8"></script>
    <script>
    const wfiFootprints = {wfi_footprints_json};

    let aladin = null, aladinReady = false;
    let showLabels = true, showGuideStars = true, surveyIdx = 0;
    let activeLayers = {{}};
    let visitColorIdx = 0;

    const VISIT_COLORS = [
        '#ffffff','#3498db','#e74c3c','#2ecc71','#f1c40f','#e67e22',
        '#9b59b6','#1abc9c','#fd79a8','#00cec9','#6c5ce7','#ff7675'
    ];
    const SURVEYS      = ['P/DSS2/color','P/2MASS/color','P/allWISE/color',
                          'P/PanSTARRS/DR1/color-z-zg-g','CDS/P/unWISE/color-W2-W1W2-W1'];
    const SURVEY_NAMES = ['DSS2','2MASS','WISE','PanSTARRS','unWISE'];

    function initAladin() {{
        if (aladinReady) return;
        A.init.then(() => {{
            aladin = A.aladin('#aladin-container', {{
                survey: SURVEYS[0], fov: 0.85, showReticle: true,
                showZoomControl: true, showLayersControl: false,
                showGotoControl: false, showFrame: true,
                cooFrame: 'J2000', projection: 'TAN'
            }});
            aladinReady = true;
        }});
    }}

    function openAladinPanel() {{
        document.getElementById('aladin-panel').style.display = 'flex';
        document.getElementById('aladin-open-btn').style.display = 'none';
        if (!aladinReady) initAladin();
        setTimeout(() => {{ if (aladin && aladin.setSize) aladin.setSize(); }}, 120);        
    }}
    function closeAladinPanel() {{
        document.getElementById('aladin-panel').style.display = 'none';
        document.getElementById('aladin-open-btn').style.display = 'flex';
    }}
    function toggleAladinExpand() {{
        const p = document.getElementById('aladin-panel');
        p.classList.toggle('expanded');
        document.getElementById('expand-btn').innerHTML =
            p.classList.contains('expanded') ? '&#x29e1;' : '&#x29e2;';
        setTimeout(() => {{ if (aladin && aladin.setSize) aladin.setSize(); }}, 120);        
    }}
    function toggleAladinSurvey() {{
        if (!aladin) return;
        surveyIdx = (surveyIdx + 1) % SURVEYS.length;
        aladin.setImageSurvey(SURVEYS[surveyIdx]);
        document.getElementById('aladin-survey-label').textContent = SURVEY_NAMES[surveyIdx];
    }}
    
    
    function redrawAll() {{
        const savedKeys = {{}};
        for (const [fpKey, layers] of Object.entries(activeLayers)) {{
            savedKeys[fpKey] = layers.info;
        }}
        clearAllFootprints();
        const savedIdx = visitColorIdx;
        visitColorIdx = 0;
        for (const [fpKey, info] of Object.entries(savedKeys)) {{
            const row = document.querySelector('tr[data-fp-key="' + fpKey + '"]');
            const visitId = row ? (row.dataset.visitId || '') : '';
            const sciId   = row ? (row.dataset.sciId || '') : '';
            addFootprint(fpKey, info.ra, info.dec, info.pa, visitId, sciId);
        }}
        updateInfoBar();
    }}

    function toggleSCALabels() {{
        showLabels = !showLabels;
        document.getElementById('label-toggle-btn').style.opacity = showLabels ? 1 : 0.4;
        redrawAll();
    }}

    function toggleGuideStars() {{
        showGuideStars = !showGuideStars;
        document.getElementById('gs-toggle-btn').style.opacity = showGuideStars ? 1 : 0.4;
        redrawAll();
    }}
    
    function removeFootprint(fpKey) {{
        if (!activeLayers[fpKey]) return;
        try {{ aladin.removeLayer(activeLayers[fpKey].overlay); }} catch(e) {{}}
        try {{ if (activeLayers[fpKey].labelOverlay) aladin.removeLayer(activeLayers[fpKey].labelOverlay); }} catch(e) {{}}
        try {{ if (activeLayers[fpKey].gsOverlay) aladin.removeLayer(activeLayers[fpKey].gsOverlay); }} catch(e) {{}}
        delete activeLayers[fpKey];
        const row = document.querySelector('tr[data-fp-key="' + fpKey + '"]');
        if (row) row.classList.remove('aladin-active');
        updateInfoBar();
    }}

    function clearAllFootprints() {{
        for (const fpKey of Object.keys(activeLayers)) {{
            try {{ aladin.removeLayer(activeLayers[fpKey].overlay); }} catch(e) {{}}
            try {{ if (activeLayers[fpKey].labelOverlay) aladin.removeLayer(activeLayers[fpKey].labelOverlay); }} catch(e) {{}}
            try {{ if (activeLayers[fpKey].gsOverlay) aladin.removeLayer(activeLayers[fpKey].gsOverlay); }} catch(e) {{}}
        }}
        activeLayers = {{}};
        visitColorIdx = 0;
        document.querySelectorAll('tr.aladin-active').forEach(el => el.classList.remove('aladin-active'));
        updateInfoBar();
    }}

    function selectAllFootprints() {{
        openAladinPanel();
        const go = () => {{
            if (!aladinReady) {{ setTimeout(go, 200); return; }}
            clearAllFootprints();
            document.querySelectorAll('tr[data-fp-key]').forEach(function(row) {{
                const fpKey   = row.dataset.fpKey;
                const ra      = parseFloat(row.dataset.ra);
                const dec     = parseFloat(row.dataset.dec);
                const pa      = parseFloat(row.dataset.pa);
                const visitId = row.dataset.visitId || '';
                const sciId   = row.dataset.sciId   || '';
                if (fpKey && !activeLayers[fpKey] && !isNaN(ra) && !isNaN(dec)) {{
                    addFootprint(fpKey, ra, dec, isNaN(pa) ? 0 : pa, visitId, sciId);
                }}
            }});
            let raMin = 999, raMax = -999, decMin = 999, decMax = -999;
            for (const fpKey of Object.keys(activeLayers)) {{
                const fp = wfiFootprints[fpKey];
                if (fp && fp.scas) {{
                    for (const corners of Object.values(fp.scas)) {{
                        for (const [r, d] of corners) {{
                            if (r < raMin)  raMin  = r;
                            if (r > raMax)  raMax  = r;
                            if (d < decMin) decMin = d;
                            if (d > decMax) decMax = d;
                        }}
                    }}
                }}
            }}
            if (raMin < 999) {{
                const cra  = (raMin + raMax) / 2;
                const cdec = (decMin + decMax) / 2;
                const span = Math.max(raMax - raMin, decMax - decMin);
                aladin.gotoRaDec(cra, cdec);
                aladin.setFoV(Math.max(span * 1.3, 0.85));
            }}
            updateInfoBar();
        }};
        go();
    }}

    function updateInfoBar() {{
        const keys = Object.keys(activeLayers);
        const n = keys.length;
        if (n === 0) {{
            document.getElementById('aladin-visit-id').textContent = '\\u2014';
            document.getElementById('aladin-ra').textContent  = '\\u2014';
            document.getElementById('aladin-dec').textContent = '\\u2014';
            document.getElementById('aladin-pa').textContent  = '\\u2014';
            document.getElementById('aladin-title').textContent = 'WFI Sky Viewer';
        }} else if (n === 1) {{
            const info = activeLayers[keys[0]].info;
            document.getElementById('aladin-visit-id').textContent = info.label;
            document.getElementById('aladin-ra').textContent  = info.ra.toFixed(5);
            document.getElementById('aladin-dec').textContent = info.dec.toFixed(5);
            document.getElementById('aladin-pa').textContent  = info.pa.toFixed(2);
            document.getElementById('aladin-title').textContent = info.label + ' \\u2014 PA ' + info.pa.toFixed(1) + '\\u00b0';
        }} else {{
            document.getElementById('aladin-visit-id').textContent = n + ' selected';
            document.getElementById('aladin-ra').textContent  = '\\u2014';
            document.getElementById('aladin-dec').textContent = '\\u2014';
            document.getElementById('aladin-pa').textContent  = '\\u2014';
            document.getElementById('aladin-title').textContent = n + ' footprints selected';
        }}
    }}

    function addFootprint(fpKey, displayRA, displayDec, displayPA, visitId, sciId) {{
        const fp = wfiFootprints[fpKey];
        const label = sciId ? ('Visit ' + visitId + ' / Exp ' + sciId) : ('Visit ' + visitId);
        const visitColor = VISIT_COLORS[visitColorIdx % VISIT_COLORS.length];
        visitColorIdx++;

        // Layer 1: SCA polygons (bottom)
        const overlay = A.graphicOverlay({{ color: visitColor, lineWidth: 2 }});
        aladin.addOverlay(overlay);

        // Layer 2: SCA labels + boresight markers (middle, on top of polygons)
        const labelOverlay = showLabels ? A.graphicOverlay({{ color: '#f1c40f', lineWidth: 1 }}) : null;
        if (labelOverlay) aladin.addOverlay(labelOverlay);

        // Layer 3: Guide stars (top)
        const gsOverlay = showGuideStars ? A.graphicOverlay({{ color: '#f1c40f', lineWidth: 2 }}) : null;
        if (gsOverlay) aladin.addOverlay(gsOverlay);

        if (fp && fp.scas) {{
            for (const [scaName, corners] of Object.entries(fp.scas)) {{
                // Filled polygon
                overlay.add(A.polygon(corners, {{
                    color: visitColor, lineWidth: 1.5,
                    fill: true, fillColor: visitColor, opacity: 0.08
                }}));
                // Outline
                overlay.add(A.polygon(corners, {{
                    color: visitColor, lineWidth: 2, fill: false
                }}));

                // SCA label marker (small circle at centroid)
                if (labelOverlay) {{
                    const cRa  = corners.reduce((s,c) => s + c[0], 0) / corners.length;
                    const cDec = corners.reduce((s,c) => s + c[1], 0) / corners.length;
                    labelOverlay.add(A.circle(cRa, cDec, 0.001, {{
                        color: '#f1c40f', lineWidth: 1
                    }}));
                }}
            }}

            // ── CGI aperture (magenta) ──
            if (fp.cgi) {{
                overlay.add(A.polygon(fp.cgi, {{
                    color: '#ff00ff', lineWidth: 1.5,
                    fill: true, fillColor: '#ff00ff', opacity: 0.06
                }}));
                overlay.add(A.polygon(fp.cgi, {{
                    color: '#ff00ff', lineWidth: 2, fill: false
                }}));
                if (labelOverlay) {{
                    const cgiRa  = fp.cgi.reduce((s,c) => s + c[0], 0) / fp.cgi.length;
                    const cgiDec = fp.cgi.reduce((s,c) => s + c[1], 0) / fp.cgi.length;
                    labelOverlay.add(A.circle(cgiRa, cgiDec, 0.001, {{
                        color: '#ff00ff', lineWidth: 1
                    }}));
                }}
            }}
            
            // V1 boresight marker
            if (fp.ra && fp.dec) {{
                const boresightOverlay = labelOverlay || overlay;
                boresightOverlay.add(A.circle(fp.ra, fp.dec, 0.002, {{
                    color: '#ffffff', lineWidth: 2
                }}));
            }}

            // WFI_CEN marker
            if (fp.ra_cen && fp.dec_cen) {{
                const cenOverlay = labelOverlay || overlay;
                cenOverlay.add(A.circle(fp.ra_cen, fp.dec_cen, 0.002, {{
                    color: '#f1c40f', lineWidth: 2
                }}));
            }}

            // Guide stars
            if (fp.guide_stars && fp.guide_stars.length > 0 && gsOverlay) {{
                for (const gs of fp.guide_stars) {{
                    const isGuide = (gs.mode === 'GUIDE');
                    const gsColor = isGuide ? '#f1c40f' : '#7f8c8d';
                    gsOverlay.add(A.circle(gs.ra, gs.dec, 0.003, {{
                        color: gsColor, lineWidth: isGuide ? 2.5 : 1.5
                    }}));
                }}
            }}
        }} else {{
            // No footprint data — just mark the pointing
            const markerOverlay = labelOverlay || overlay;
            markerOverlay.add(A.circle(displayRA, displayDec, 0.003, {{
                color: '#e74c3c', lineWidth: 2
            }}));
        }}

        activeLayers[fpKey] = {{
            overlay: overlay,
            labelOverlay: labelOverlay,
            gsOverlay: gsOverlay,
            info: {{ label: label, ra: displayRA, dec: displayDec, pa: displayPA }}
        }};

        const row = document.querySelector('tr[data-fp-key="' + fpKey + '"]');
        if (row) row.classList.add('aladin-active');
    }}

    function showFootprint(fpKey, displayRA, displayDec, displayPA, visitId, sciId, multiSelect) {{
        openAladinPanel();
        const go = () => {{
            if (!aladinReady) {{ setTimeout(go, 200); return; }}
            if (multiSelect) {{
                if (activeLayers[fpKey]) {{
                    removeFootprint(fpKey);
                }} else {{
                    addFootprint(fpKey, displayRA, displayDec, displayPA, visitId, sciId);
                }}
            }} else {{
                clearAllFootprints();
                addFootprint(fpKey, displayRA, displayDec, displayPA, visitId, sciId);
            }}
            const fp = wfiFootprints[fpKey];
            const cra  = (fp && fp.ra_cen) ? fp.ra_cen : displayRA;
            const cdec = (fp && fp.dec_cen) ? fp.dec_cen : displayDec;
            if (!multiSelect || Object.keys(activeLayers).length <= 1) {{
                aladin.gotoRaDec(cra, cdec);
                aladin.setFoV(0.85);
            }}
            updateInfoBar();
        }};
        go();
    }}

    document.addEventListener('DOMContentLoaded', function() {{
        document.getElementById('aladin-open-btn').style.display = 'flex';
        document.querySelectorAll('tr[data-fp-key]').forEach(function(row) {{
            row.addEventListener('click', function(e) {{
                if (e.target.tagName === 'A' || e.target.tagName === 'BUTTON' ||
                    e.target.closest('.expand-btn') ||
                    e.target.closest('.sky-preview-wrapper')) return;
                const fpKey   = this.dataset.fpKey;
                const ra      = parseFloat(this.dataset.ra);
                const dec     = parseFloat(this.dataset.dec);
                const pa      = parseFloat(this.dataset.pa);
                const visitId = this.dataset.visitId  || '';
                const sciId   = this.dataset.sciId    || '';
                const multi   = e.ctrlKey || e.metaKey;
                if (fpKey && !isNaN(ra) && !isNaN(dec)) {{
                    showFootprint(fpKey, ra, dec, isNaN(pa) ? 0 : pa, visitId, sciId, multi);
                }}
            }});
        }});
        (function() {{
            const panel  = document.getElementById('aladin-panel');
            const header = document.getElementById('aladin-panel-header');
            let drag = false, sx, sy, ox, oy;
            header.addEventListener('mousedown', function(e) {{
                if (e.target.tagName === 'BUTTON') return;
                drag = true; sx = e.clientX; sy = e.clientY;
                const r = panel.getBoundingClientRect();
                ox = r.left; oy = r.top;
                panel.style.left = ox+'px'; panel.style.top = oy+'px';
                panel.style.right = 'auto'; panel.style.bottom = 'auto';
            }});
            document.addEventListener('mousemove', function(e) {{
                if (!drag) return;
                panel.style.left = (ox + e.clientX - sx) + 'px';
                panel.style.top  = (oy + e.clientY - sy) + 'px';
            }});
            document.addEventListener('mouseup', function() {{ drag = false; }});
        }})();
        if (window.ResizeObserver) {{
            new ResizeObserver(() => {{
                if (aladin) {{
                    if (aladin.setSize) {{ aladin.setSize(); }}
                    else if (aladin.view && aladin.view.requestRedraw) {{ aladin.view.requestRedraw(); }}
                }}
            }}).observe(document.getElementById('aladin-panel'));
        }}        
    }});
    </script>
"""

def _add_data_attrs_to_table_rows(html_content, df):
    """
    Post-process rendered HTML to add data-fp-key, data-ra, data-dec,
    data-pa, data-visit-id, data-sci-id attributes to <tr> elements.
    
    Strategy: Build a lookup from SCI_ID → attributes, then scan every
    <tr in the HTML. For each <tr>, look ahead in that row's content for 
    a known SCI_ID and inject the attrs.
    """
    have_radec = all(c in df.columns for c in ['RA', 'DEC', 'Position_Angle'])
    if 'Visit_ID' not in df.columns:
        return html_content

    # Build lookup: SCI_ID string → attr dict
    # SCI_ID is unique per row, so it's the best token to search for
    use_sci_id = 'SCI_ID' in df.columns
    
    token_to_attrs = {}   # search_token → HTML attr string

    for _, row in df.iterrows():
        vid = str(row.get('Visit_ID', '')).strip()
        if not vid:
            continue
        sid = str(row.get('SCI_ID', '')).strip() if use_sci_id else ''
        fp_key = f"{vid}_{sid}" if sid else vid

        # The token we'll search for in the HTML to identify this row
        search_token = sid if sid else vid
        if not search_token or search_token in token_to_attrs:
            continue

        parts = [f'data-fp-key="{fp_key}"', f'data-visit-id="{vid}"']
        if sid:
            parts.append(f'data-sci-id="{sid}"')

        try:
            ra = float(row.get('RA', float('nan')))
            if not np.isnan(ra):
                parts.append(f'data-ra="{ra}"')
        except (ValueError, TypeError):
            pass
        try:
            dec = float(row.get('DEC', float('nan')))
            if not np.isnan(dec):
                parts.append(f'data-dec="{dec}"')
        except (ValueError, TypeError):
            pass
        try:
            pa = float(row.get('Position_Angle', float('nan')))
            if not np.isnan(pa):
                parts.append(f'data-pa="{pa}"')
        except (ValueError, TypeError):
            pass

        token_to_attrs[search_token] = ' '.join(parts)

    if not token_to_attrs:
        return html_content

    # Split HTML into chunks between <tr and the next <tr (or end)
    # This avoids cross-row regex matching entirely.
    #
    # We split on every occurrence of '<tr' and process each chunk.
    
    pieces = re.split(r'(<tr\b)', html_content)
    # pieces = ['stuff before first <tr', '<tr', ' rest of row 1...', '<tr', ' rest of row 2...', ...]
    # Even indices are content between/before <tr tags
    # Odd indices are the literal '<tr' strings

    modified_count = 0
    used_tokens = set()

    for i in range(1, len(pieces), 2):
        # pieces[i] = '<tr'
        # pieces[i+1] = ' class="...">\n  <td>...</td>\n  ...\n</tr>\n...'
        if i + 1 >= len(pieces):
            break

        chunk = pieces[i + 1]

        # Skip if this <tr> already has data-fp-key
        # (check just the opening tag portion, up to first >)
        tag_end = chunk.find('>')
        if tag_end == -1:
            continue
        opening_attrs = chunk[:tag_end]
        if 'data-fp-key' in opening_attrs:
            continue

        # Search this chunk for any of our tokens
        for token, attr_str in token_to_attrs.items():
            if token in used_tokens:
                continue
            if token in chunk:
                # Inject attrs into the opening <tr tag
                pieces[i + 1] = ' ' + attr_str + chunk
                modified_count += 1
                used_tokens.add(token)
                break  # one token per row

    html_content = ''.join(pieces)
    print(f"  🏷️  Added Aladin data attributes to {modified_count} table rows")
    return html_content


def inject_aladin_into_html(html_content, wfi_footprints, df=None):
    """
    Inject Aladin Lite CSS, HTML panel, JS, AND data attributes into
    a rendered HTML string.
    
    Args:
        html_content:   str — the complete HTML report string
        wfi_footprints: dict from precompute_wfi_footprints()
        df:             pandas DataFrame (opup_info) — needed to add 
                        data-fp-key attrs to table rows
    """
    fp_json = json.dumps(wfi_footprints if wfi_footprints else {})

    # 1. Add data-fp-key attributes to <tr> elements 
    if df is not None:
        html_content = _add_data_attrs_to_table_rows(html_content, df)

    # 2. CSS → insert before the FIRST </style>
    css = get_aladin_css()
    html_content = html_content.replace('</style>', css + '\n    </style>', 1)

    # 3. HTML panel + JS → insert before </body>
    panel_html = get_aladin_html()
    panel_js   = get_aladin_javascript(fp_json)
    html_content = html_content.replace(
        '</body>',
        panel_html + '\n' + panel_js + '\n</body>',
        1
    )

    return html_content

def generate_html_report(df, opup_filepath, sky_plotter_html=None, visit_png_map=None, skyplot_mosaic_filename=None):
    """
    Generate HTML content with hyperlinks to visit files and horizontal scrolling.
    Includes embedded visit file contents with syntax highlighting.
    
    Args:
        df: DataFrame containing OPUP data
        opup_filepath: Path to the OPUP archive for creating links
        sky_plotter_html: Optional path to sky plotter HTML file for cross-linking
    
    Returns:
        HTML string
    """
    opup_name = Path(opup_filepath).name

    if visit_png_map is None:
        visit_png_map = {}
    if skyplot_mosaic_filename is None:
        skyplot_mosaic_filename = ''

    
    # Extract all visit file contents
    print("Extracting visit file contents from archive...")
    visit_filenames = df['Visit_File_Name'].dropna().unique().tolist()
    visit_contents = get_all_visit_contents(opup_filepath, visit_filenames)
    print(f"Extracted {len(visit_contents)} visit files")
    
    # Apply syntax highlighting to each visit file
    print("Applying syntax highlighting...")
    visit_contents_highlighted = {}
    for filename, content in visit_contents.items():
        visit_contents_highlighted[filename] = syntax_highlight_visit_content(content)
    
    # Convert to JSON for embedding (store both raw and highlighted)
    visit_contents_raw_json = json.dumps(visit_contents)
    visit_contents_highlighted_json = json.dumps(visit_contents_highlighted)
    
    # Calculate statistics - MUST BE BEFORE HTML STRING STARTS
    total_rows = len(df)
    total_cols = len(df.columns)
    unique_visits = df['Visit_ID'].nunique() if 'Visit_ID' in df.columns else 0
    unique_sci_ids = df['SCI_ID'].nunique() if 'SCI_ID' in df.columns else 0
    
    # Calculate program/pass/segment/observation counts
    # These are hierarchical, so we need to count unique combinations
    # Use unique visits to avoid counting duplicates from multiple exposures
    if 'Visit_ID' in df.columns:
        unique_visits_df = df.drop_duplicates(subset=['Visit_ID'])
        
        # Programs: Count unique program numbers
        unique_programs = unique_visits_df['Program_Number'].nunique() if 'Program_Number' in unique_visits_df.columns else 0
        
        # Passes: Count unique (Program, Pass) combinations
        if 'Program_Number' in unique_visits_df.columns and 'Pass_Number' in unique_visits_df.columns:
            unique_passes = unique_visits_df[['Program_Number', 'Pass_Number']].drop_duplicates().shape[0]
        else:
            unique_passes = 0
        
        # Segments: Count unique (Program, Pass, Segment) combinations
        if all(col in unique_visits_df.columns for col in ['Program_Number', 'Pass_Number', 'Segment_Number']):
            unique_segments = unique_visits_df[['Program_Number', 'Pass_Number', 'Segment_Number']].drop_duplicates().shape[0]
        else:
            unique_segments = 0
        
        # Observations: Count unique (Program, Pass, Segment, Observation) combinations
        if all(col in unique_visits_df.columns for col in ['Program_Number', 'Pass_Number', 'Segment_Number', 'Observation_Number']):
            unique_observations = unique_visits_df[['Program_Number', 'Pass_Number', 'Segment_Number', 'Observation_Number']].drop_duplicates().shape[0]
        else:
            unique_observations = 0
    else:
        unique_programs = 0
        unique_passes = 0
        unique_segments = 0
        unique_observations = 0
    
    # Calculate total duration - ONLY count once per visit, not per exposure
    total_duration_seconds = 0
    total_duration_display = "N/A"
    if 'Duration' in df.columns and 'Visit_ID' in df.columns:
        try:
            # Reuse unique_visits_df if already created
            if 'unique_visits_df' not in locals():
                unique_visits_df = df.drop_duplicates(subset=['Visit_ID'])
            # Convert to numeric, coercing errors to NaN
            durations = pd.to_numeric(unique_visits_df['Duration'], errors='coerce')
            total_duration_seconds = durations.sum()
            
            # Only create display string if we have valid data
            if pd.notna(total_duration_seconds) and total_duration_seconds > 0:
                # Convert to hours, minutes, seconds
                hours = int(total_duration_seconds // 3600)
                minutes = int((total_duration_seconds % 3600) // 60)
                seconds = int(total_duration_seconds % 60)
                total_duration_display = f"{hours}h {minutes}m {seconds}s"
            else:
                total_duration_seconds = 0
                total_duration_display = "N/A"
        except Exception as e:
            print(f"Error calculating duration: {e}")
            total_duration_seconds = 0
            total_duration_display = "N/A"
    


    # Start HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OPUP Report - {opup_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metadata p {{
            margin: 5px 0;
            font-size: 14px;
        }}
        .sky-preview-wrapper {{
            position: relative;
            display: inline-block;
        }}

        .sky-preview-tooltip {{
            display: none;
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            background: #1a1a2e;
            border: 2px solid #3498db;
            border-radius: 8px;
            padding: 4px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            pointer-events: none;
        }}

        .sky-preview-tooltip img {{
            width: 480px;
            height: auto;
            border-radius: 4px;
            display: block;
        }}

        .sky-preview-tooltip::after {{
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 8px solid transparent;
            border-top-color: #3498db;
        }}

        .sky-preview-wrapper:hover .sky-preview-tooltip {{
            display: block;
        }}

        td:nth-last-child(-n+3) .sky-preview-tooltip {{
            left: auto;
            right: 0;
            transform: none;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .summary-card:nth-child(2) {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .summary-card:nth-child(3) {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        .summary-card:nth-child(4) {{
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }}
        .summary-card:nth-child(5) {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}
        .summary-card:nth-child(6) {{
            background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        }}
        .summary-card:nth-child(7) {{
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        }}
        .summary-card:nth-child(8) {{
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        }}
        .summary-card:nth-child(9) {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        }}
        .summary-card:nth-child(10) {{
            background: linear-gradient(135deg, #ff6e7f 0%, #bfe9ff 100%);
        }}
        .summary-label {{
            font-size: 12px;
            opacity: 0.9;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .summary-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 5px 0;
        }}
        .summary-value.time {{
            font-size: 24px;
        }}
        .summary-detail {{
            font-size: 11px;
            opacity: 0.8;
            margin-top: 5px;
        }}
        .breakdown-section {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .breakdown-section h3 {{
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }}
        .breakdown-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        .breakdown-item:last-child {{
            border-bottom: none;
        }}
        .breakdown-label {{
            font-weight: 500;
            color: #34495e;
        }}
        .breakdown-value {{
            color: #7f8c8d;
            font-weight: bold;
        }}

        /* Program hierarchy styling */
        .program-header {{
            background: linear-gradient(to right, #f8f9fa 0%, #ffffff 100%);
            font-weight: 600;
            border-left: 5px solid #3498db;
            padding: 12px 15px !important;
            margin-bottom: 5px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .program-badge {{
            display: inline-block;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 13px;
            font-weight: 700;
            margin-right: 10px;
            letter-spacing: 0.5px;
        }}
        .purpose-tag {{
            display: inline-block;
            background-color: #ecf0f1;
            color: #34495e;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 500;
            margin-left: 8px;
            font-style: italic;
        }}
        .purpose-tag-small {{
            display: inline-block;
            background-color: #e8f4f8;
            color: #2980b9;
            padding: 2px 6px;
            border-radius: 2px;
            font-size: 10px;
            font-weight: 500;
            margin-left: 5px;
            font-style: italic;
        }}
        .pass-item {{
            padding: 8px 15px 8px 40px !important;
            background-color: #fafafa;
            border-left: 3px solid #95a5a6;
            margin-left: 15px;
            margin-bottom: 3px;
            border-radius: 0 3px 3px 0;
        }}
        .segment-item {{
            padding: 6px 15px 6px 60px !important;
            font-size: 11px;
            color: #7f8c8d;
            font-style: italic;
            background-color: #f5f5f5;
            margin-left: 15px;
            border-radius: 0 3px 3px 0;
        }}

        .table-wrapper {{
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .table-container {{
            overflow-x: auto;
            overflow-y: auto;
            max-height: 700px;
            position: relative;
        }}
        /* Custom scrollbar styling */
        .table-container::-webkit-scrollbar {{
            height: 12px;
            width: 12px;
        }}
        .table-container::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 10px;
        }}
        .table-container::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 10px;
        }}
        .table-container::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
        table {{
            border-collapse: collapse;
            width: max-content;
            min-width: 100%;
            font-size: 12px;
        }}
        thead {{
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px 15px;
            text-align: left;
            white-space: nowrap;
            border-right: 1px solid #2980b9;
            font-weight: 600;
        }}
        th:last-child {{
            border-right: none;
        }}
        /* Make first few columns sticky */
        th:nth-child(1),
        td:nth-child(1) {{
            position: sticky;
            left: 0;
            z-index: 10;
            background-color: #fff;
        }}
        th:nth-child(1) {{
            background-color: #2980b9;
            z-index: 101;
        }}
        th:nth-child(2),
        td:nth-child(2) {{
            position: sticky;
            left: 0;
            z-index: 9;
            background-color: #fff;
        }}
        th:nth-child(2) {{
            background-color: #2980b9;
            z-index: 101;
        }}
        td {{
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
            border-right: 1px solid #eee;
            white-space: nowrap;
        }}
        td:last-child {{
            border-right: none;
        }}
        tr:hover td {{
            background-color: #e8f4f8;
        }}
        tr:nth-child(even) td {{
            background-color: #f9f9f9;
        }}
        tr:nth-child(even):hover td {{
            background-color: #e8f4f8;
        }}
        /* Keep sticky columns highlighted on hover */
        tr:hover td:nth-child(1),
        tr:hover td:nth-child(2) {{
            background-color: #d5e8f3;
        }}
        a.visit-link {{
            color: #2980b9;
            text-decoration: none;
            font-weight: bold;
            cursor: pointer;
        }}
        a.visit-link:hover {{
            color: #3498db;
            text-decoration: underline;
        }}
        .highlight {{
            background-color: #fff9c4;
            font-weight: 600;
        }}
        .scroll-indicator {{
            text-align: center;
            padding: 10px;
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            margin-bottom: 10px;
            font-size: 13px;
        }}
        .scroll-indicator::before {{
            content: "⟷ ";
            font-weight: bold;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin-top: 10px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }}
        
        /* Modal styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.7);
        }}
        .modal-content {{
            background-color: #fefefe;
            margin: 1% auto;
            padding: 0;
            border: 1px solid #888;
            width: 96%;
            max-width: 1800px;
            height: 94vh;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
        }}
        .modal-header {{
            padding: 15px 20px;
            background-color: #3498db;
            color: white;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }}
        .modal-header h2 {{
            margin: 0;
            font-size: 18px;
        }}
        .modal-controls {{
            display: flex;
            gap: 15px;
            align-items: center;
        }}
        .syntax-toggle {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}
        .syntax-toggle input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        .close {{
            color: white;
            font-size: 32px;
            font-weight: bold;
            cursor: pointer;
            line-height: 20px;
            padding: 0 10px;
        }}
        .close:hover,
        .close:focus {{
            color: #ecf0f1;
        }}
        .modal-body {{
            padding: 0;
            overflow: hidden;
            flex: 1;
            display: flex;
            flex-direction: column;
        }}
        .visit-content {{
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            font-family: 'Consolas', 'Courier New', Monaco, monospace;
            font-size: 11px;
            white-space: pre;
            line-height: 1.5;
            overflow: auto;
            flex: 1;
            margin: 0;
        }}
        
        /* STOL Syntax Highlighting Colors */
        .stol-keyword {{
            color: #c586c0;
            font-weight: bold;
        }}
        .stol-function {{
            color: #dcdcaa;
            font-weight: bold;
        }}
        .stol-number {{
            color: #b5cea8;
        }}
        .stol-string {{
            color: #ce9178;
        }}
        .stol-comment {{
            color: #6a9955;
            font-style: italic;
        }}
        .stol-constant {{
            color: #4fc1ff;
            font-weight: 600;
        }}
        .stol-variable {{
            color: #9cdcfe;
        }}
        .stol-param {{
            color: #f48771;
        }}
        .stol-param-name {{
            color: #9cdcfe;
        }}
        .stol-punctuation {{
            color: #d4d4d4;
        }}
        .stol-brace {{
            color: #ffd700;
            font-weight: bold;
        }}
        
        /* Custom scrollbar for visit content */
        .visit-content::-webkit-scrollbar {{
            height: 14px;
            width: 14px;
        }}
        .visit-content::-webkit-scrollbar-track {{
            background: #0d0d0d;
            border-radius: 10px;
        }}
        .visit-content::-webkit-scrollbar-thumb {{
            background: #4a4a4a;
            border-radius: 10px;
        }}
        .visit-content::-webkit-scrollbar-thumb:hover {{
            background: #5a5a5a;
        }}
        .modal-footer {{
            padding: 12px 20px;
            background-color: #ecf0f1;
            border-radius: 0 0 8px 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }}
        .footer-info {{
            font-size: 12px;
            color: #7f8c8d;
        }}
        .footer-buttons {{
            display: flex;
            gap: 10px;
        }}
        .download-btn {{
            background-color: #27ae60;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        }}
        .download-btn:hover {{
            background-color: #229954;
        }}
        .copy-btn {{
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        }}
        .copy-btn:hover {{
            background-color: #2980b9;
        }}
        .scroll-hint {{
            position: absolute;
            top: 80px;
            right: 40px;
            background-color: rgba(255, 193, 7, 0.9);
            color: #333;
            padding: 8px 15px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            animation: fadeOut 4s forwards;
        }}
        @keyframes fadeOut {{
            0% {{ opacity: 1; }}
            70% {{ opacity: 1; }}
            100% {{ opacity: 0; }}
        }}

        /* Collapsible observation group styles */
        .group-header-row {{
            cursor: pointer;
        }}
        .group-header-row td:first-child {{
            position: relative;
        }}
        .expand-btn {{
            display: inline-block;
            width: 20px;
            height: 20px;
            line-height: 20px;
            text-align: center;
            background-color: #3498db;
            color: white;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
            cursor: pointer;
            margin-right: 5px;
            user-select: none;
            vertical-align: middle;
        }}
        .expand-btn:hover {{
            background-color: #2980b9;
        }}
        .expand-btn.expanded {{
            background-color: #e74c3c;
        }}
        .group-child-row {{
            display: none;
            background-color: #f9f9f9 !important;
        }}
        .group-child-row.visible {{
            display: table-row;
        }}
        .group-child-row td {{
            border-top: 1px dashed #ddd !important;
        }}
        .obs-count-badge {{
            display: inline-block;
            background-color: #e67e22;
            color: white;
            padding: 1px 6px;
            border-radius: 10px;
            font-size: 10px;
            margin-left: 5px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <h1>OPUP Report: {opup_name}</h1>

    <div class="metadata">
        <p><strong>Archive:</strong> {opup_name}</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
    
    # Add sky plotter link if available
    if sky_plotter_html:
        sky_plotter_filename = os.path.basename(sky_plotter_html)
        html += f"""        <p><strong>🌌 <a href="{sky_plotter_filename}" target="_blank" style="color: #3498db; text-decoration: none; font-weight: bold;">Sky Map →</a></strong></p>
"""
    
    # DEBUG — remove after confirming
    print(f"  DEBUG: skyplot_mosaic_filename = '{skyplot_mosaic_filename}'")

        # Add sky plot mosaic link if available    
    if skyplot_mosaic_filename:
        html += f"""        <p><strong>🔭 <a href="{skyplot_mosaic_filename}" target="_blank" style="color: #3498db; text-decoration: none; font-weight: bold;">Visit Sky Plots →</a></strong></p>
"""

    html += f"""    </div>


    
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-label">Programs</div>
            <div class="summary-value">{unique_programs}</div>
            <div class="summary-detail">Unique program numbers</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Passes</div>
            <div class="summary-value">{unique_passes}</div>
            <div class="summary-detail">Unique pass numbers</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Segments</div>
            <div class="summary-value">{unique_segments}</div>
            <div class="summary-detail">Unique segment numbers</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Observations</div>
            <div class="summary-value">{unique_observations}</div>
            <div class="summary-detail">Unique observation numbers</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Visits</div>
            <div class="summary-value">{unique_visits}</div>
            <div class="summary-detail">Distinct Visit IDs</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Total Exposures</div>
            <div class="summary-value">{total_rows}</div>
            <div class="summary-detail">All rows in data</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Total Duration</div>
            <div class="summary-value time">{total_duration_display}</div>
            <div class="summary-detail">{int(total_duration_seconds):,} seconds</div>
        </div>
    </div>
"""
# Add Program/Pass/Segment hierarchy breakdown
    if 'Visit_ID' in df.columns:
        unique_visits_df = df.drop_duplicates(subset=['Visit_ID'])
        
        # Program breakdown
        if 'Program_Number' in unique_visits_df.columns:
            html += """
    <div class="breakdown-section">
        <h3>📋 Program Hierarchy</h3>
"""
            # Define color palette for programs
            program_colors = [
                '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b'
            ]
            
            # Group by Program
            if 'Duration' in unique_visits_df.columns:
                unique_visits_df_copy = unique_visits_df.copy()
                unique_visits_df_copy['Duration'] = pd.to_numeric(unique_visits_df_copy['Duration'], errors='coerce')
                
                prog_breakdown = unique_visits_df_copy.groupby('Program_Number').agg({
                    'Visit_ID': 'count',
                    'Duration': 'sum'
                }).reset_index()
                prog_breakdown.columns = ['Program', 'Visits', 'Duration']
            else:
                prog_breakdown = unique_visits_df.groupby('Program_Number').agg({
                    'Visit_ID': 'count'
                }).reset_index()
                prog_breakdown.columns = ['Program', 'Visits']
            
            prog_breakdown = prog_breakdown.sort_values('Program')
            
            # Display each program
            for prog_idx, prog_row in enumerate(prog_breakdown.iterrows()):
                _, prog_row = prog_row
                prog_num = prog_row['Program']
                prog_visits = int(prog_row['Visits'])
                
                # Assign color to program
                prog_color = program_colors[prog_idx % len(program_colors)]
                
                # Format duration if available
                duration_str = ""
                if 'Duration' in prog_breakdown.columns and prog_row['Duration'] > 0:
                    hours = int(prog_row['Duration'] // 3600)
                    minutes = int((prog_row['Duration'] % 3600) // 60)
                    duration_str = f" | {hours}h {minutes}m"
                
                # Get intended purpose for this program if available
                prog_purpose = ""
                if 'Intended_Purpose' in unique_visits_df.columns:
                    prog_data = unique_visits_df[unique_visits_df['Program_Number'] == prog_num]
                    purposes = prog_data['Intended_Purpose'].dropna().unique()
                    if len(purposes) > 0:
                        # Take first purpose or most common if multiple
                        prog_purpose = purposes[0]
                        if len(purposes) > 1:
                            prog_purpose += f" (+{len(purposes)-1} more)"
                
                html += f"""        <div class="breakdown-item program-header" style="border-left-color: {prog_color};">
            <span class="breakdown-label">
                <span class="program-badge" style="background-color: {prog_color};">Program {prog_num}</span>
"""
                
                if prog_purpose:
                    html += f"""                <span class="purpose-tag">{prog_purpose}</span>
"""
                
                html += f"""            </span>
            <span class="breakdown-value">{prog_visits} visits{duration_str}</span>
        </div>
"""
                
                # Get passes for this program
                if 'Pass_Number' in unique_visits_df.columns:
                    prog_data = unique_visits_df[unique_visits_df['Program_Number'] == prog_num]
                    
                    if 'Duration' in prog_data.columns:
                        prog_data_copy = prog_data.copy()
                        prog_data_copy['Duration'] = pd.to_numeric(prog_data_copy['Duration'], errors='coerce')
                        
                        pass_breakdown = prog_data_copy.groupby('Pass_Number').agg({
                            'Visit_ID': 'count',
                            'Duration': 'sum'
                        }).reset_index()
                        pass_breakdown.columns = ['Pass', 'Visits', 'Duration']
                    else:
                        pass_breakdown = prog_data.groupby('Pass_Number').agg({
                            'Visit_ID': 'count'
                        }).reset_index()
                        pass_breakdown.columns = ['Pass', 'Visits']
                    
                    pass_breakdown = pass_breakdown.sort_values('Pass')
                    
                    # Display passes under this program
                    for _, pass_row in pass_breakdown.iterrows():
                        pass_num = pass_row['Pass']
                        pass_visits = int(pass_row['Visits'])
                        
                        # Format duration if available
                        pass_duration_str = ""
                        if 'Duration' in pass_breakdown.columns and pass_row['Duration'] > 0:
                            hours = int(pass_row['Duration'] // 3600)
                            minutes = int((pass_row['Duration'] % 3600) // 60)
                            pass_duration_str = f" | {hours}h {minutes}m"
                        
                        # Get intended purposes for this pass
                        pass_purposes = ""
                        if 'Intended_Purpose' in unique_visits_df.columns:
                            pass_data = prog_data[prog_data['Pass_Number'] == pass_num]
                            purposes = pass_data['Intended_Purpose'].dropna().unique()
                            if len(purposes) > 0:
                                # Show all unique purposes for this pass
                                if len(purposes) == 1:
                                    pass_purposes = f' <span class="purpose-tag-small">{purposes[0]}</span>'
                                else:
                                    pass_purposes = f' <span class="purpose-tag-small">{len(purposes)} purposes</span>'
                        
                        html += f"""        <div class="breakdown-item pass-item" style="border-left-color: {prog_color};">
            <span class="breakdown-label">└─ Pass {pass_num}{pass_purposes}</span>
            <span class="breakdown-value">{pass_visits} visits{pass_duration_str}</span>
        </div>
"""
                        
                        # Get segments for this pass (if available)
                        if 'Segment_Number' in unique_visits_df.columns:
                            pass_data = prog_data[prog_data['Pass_Number'] == pass_num]
                            segments = pass_data['Segment_Number'].unique()
                            segments = sorted([s for s in segments if pd.notna(s)])
                            
                            if len(segments) > 0:
                                if len(segments) <= 5:
                                    segment_list = ', '.join([f"S{int(s):03d}" for s in segments])
                                else:
                                    segment_list = ', '.join([f"S{int(s):03d}" for s in segments[:5]]) + f" ... (+{len(segments)-5} more)"
                                
                                html += f"""        <div class="breakdown-item segment-item">
            <span class="breakdown-label">   └─ Segments: {segment_list}</span>
            <span class="breakdown-value">{len(segments)} total</span>
        </div>
"""
            
            html += "    </div>\n"

    # Add instrument breakdown if available
    if 'Science_Instrument' in df.columns:
        html += """
    <div class="breakdown-section">
        <h3>Instrument Breakdown</h3>
"""
        # Create aggregation dictionary
        agg_dict = {
            'Visit_ID': 'nunique',
            'SCI_ID': 'count'
        }
        
        instr_breakdown = df.groupby('Science_Instrument', as_index=False).agg(agg_dict)
        instr_breakdown.columns = ['Instrument', 'Unique Visits', 'Total Exposures']
        
        # Add duration if available - only count once per visit
        if 'Duration' in df.columns and 'Visit_ID' in df.columns:
            try:
                # Get unique visits only for duration calculation
                unique_visits_df = df.drop_duplicates(subset=['Visit_ID']).copy()
                # Convert duration to numeric
                unique_visits_df['Duration'] = pd.to_numeric(unique_visits_df['Duration'], errors='coerce')
                instr_duration = unique_visits_df.groupby('Science_Instrument', as_index=False)['Duration'].sum()
                instr_duration.columns = ['Instrument', 'Total Duration']
                instr_breakdown = instr_breakdown.merge(instr_duration, on='Instrument', how='left')
                instr_breakdown['Total Duration'] = instr_breakdown['Total Duration'].fillna(0)
                
                for _, row in instr_breakdown.iterrows():
                    if row['Total Duration'] > 0:
                        hours = int(row['Total Duration'] // 3600)
                        minutes = int((row['Total Duration'] % 3600) // 60)
                        duration_str = f" | {hours}h {minutes}m"
                    else:
                        duration_str = ""
                    html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Instrument']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures{duration_str}</span>
        </div>
"""
            except Exception as e:
                print(f"Error calculating instrument duration: {e}")
                for _, row in instr_breakdown.iterrows():
                    html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Instrument']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures</span>
        </div>
"""
        else:
            # Duration column not available, display without duration
            for _, row in instr_breakdown.iterrows():
                html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Instrument']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures</span>
        </div>
"""
        html += "    </div>\n"
    
    # Add filter breakdown if available
    if 'WFI_Optical_Element' in df.columns:
        html += """
    <div class="breakdown-section">
        <h3>Filter Usage</h3>
"""
        # Create aggregation dictionary
        agg_dict = {
            'Visit_ID': 'nunique',
            'SCI_ID': 'count'
        }
        
        filter_breakdown = df.groupby('WFI_Optical_Element', as_index=False).agg(agg_dict)
        filter_breakdown.columns = ['Filter', 'Unique Visits', 'Total Exposures']
        
        # Add duration if available - only count once per visit
        if 'Duration' in df.columns and 'Visit_ID' in df.columns:
            try:
                # Get unique visits only for duration calculation
                unique_visits_df = df.drop_duplicates(subset=['Visit_ID']).copy()
                # Convert duration to numeric
                unique_visits_df['Duration'] = pd.to_numeric(unique_visits_df['Duration'], errors='coerce')
                filter_duration = unique_visits_df.groupby('WFI_Optical_Element', as_index=False)['Duration'].sum()
                filter_duration.columns = ['Filter', 'Total Duration']
                filter_breakdown = filter_breakdown.merge(filter_duration, on='Filter', how='left')
                filter_breakdown['Total Duration'] = filter_breakdown['Total Duration'].fillna(0)
                filter_breakdown = filter_breakdown.sort_values('Total Duration', ascending=False)
                
                for _, row in filter_breakdown.iterrows():
                    if pd.notna(row['Filter']):
                        if row['Total Duration'] > 0:
                            hours = int(row['Total Duration'] // 3600)
                            minutes = int((row['Total Duration'] % 3600) // 60)
                            duration_str = f" | {hours}h {minutes}m"
                        else:
                            duration_str = ""
                        html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Filter']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures{duration_str}</span>
        </div>
"""
            except Exception as e:
                print(f"Error calculating filter duration: {e}")
                filter_breakdown = filter_breakdown.sort_values('Total Exposures', ascending=False)
                for _, row in filter_breakdown.iterrows():
                    if pd.notna(row['Filter']):
                        html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Filter']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures</span>
        </div>
"""
        else:
            # Duration column not available, display without duration
            filter_breakdown = filter_breakdown.sort_values('Total Exposures', ascending=False)
            for _, row in filter_breakdown.iterrows():
                if pd.notna(row['Filter']):
                    html += f"""        <div class="breakdown-item">
            <span class="breakdown-label">{row['Filter']}</span>
            <span class="breakdown-value">{int(row['Unique Visits'])} visits | {int(row['Total Exposures'])} exposures</span>
        </div>
"""
        html += "    </div>\n"
        
    # Determine if grouping is possible (needed for both buttons and row generation)
    GROUP_COLS = ['Program_Number', 'Exec_Plan_Number', 'Pass_Number', 
                  'Segment_Number', 'Observation_Number']
    group_cols_present = all(col in df.columns for col in GROUP_COLS)
        # Add expand/collapse controls above the table
    if group_cols_present:
        html += """
    <div class="group-controls">
        <button class="expand-all-btn" onclick="toggleAllVisitGroups(true)">▼ Expand All Visits</button>
        <button class="collapse-all-btn" onclick="toggleAllVisitGroups(false)">▶ Collapse All Visits</button>
    </div>
"""

    html += f"""
    <div class="scroll-indicator">
        Scroll horizontally to view all {len(df.columns)} columns | First columns are fixed for easy reference | Click visit file names to view content
    </div>
    
    <div class="table-wrapper">
        <div class="table-container">
            <table>
                <thead>
                    <tr>
"""
    
    # Add table headers
    for col in df.columns:
        html += f"                        <th>{col}</th>\n"
    
    html += """                    </tr>
                </thead>
                <tbody>
"""
    
# ================================================================
    # Add table rows with visit-level grouping/collapsing
    # ================================================================
    # Group by Program + Exec Plan + Pass + Segment + Observation
    # Rows sharing these 5 columns are different exposures of the same visit
    # GROUP_COLS = ['Program_Number', 'Exec_Plan_Number', 'Pass_Number', 
    #               'Segment_Number', 'Observation_Number']
    
    # group_cols_present = all(col in df.columns for col in GROUP_COLS)
    
    # Helper function to format a single cell
    def _format_cell(col, value):
        if pd.isna(value):
            return ""
        elif col in ['Visit_ID', 'SCI_ID']:
            return f'<span class="highlight">{html_module.escape(str(value))}</span>'
        elif col == 'Visit_File_Name' and str(value).endswith('.vst'):
            vf = str(value)
            png_path = visit_png_map.get(vf, '')

            # Existing: click visit name to view STOL content
            link_html = (
                f'<a href="#" class="visit-link" '
                f'onclick="showVisitContent(\'{vf}\'); return false;" '
                f'title="Click to view {vf}">{vf}</a>'
            )

            if png_path and skyplot_mosaic_filename:
                anchor = vf.replace('.vst', '')
                link_html += (
                    f' <a href="{skyplot_mosaic_filename}#{anchor}" '
                    f'target="_blank" title="View sky plot" '
                    f'style="text-decoration:none;">🔭</a>'
                )

            return link_html
        else:
            return html_module.escape(str(value))


            
    if group_cols_present:
        # Build group key per row and identify groups
        df_temp = df.copy()
        df_temp['_group_key'] = df_temp[GROUP_COLS].astype(str).agg('|'.join, axis=1)
        
        # Count rows per group
        group_sizes = df_temp.groupby('_group_key', sort=False).size()
        
        # Track group transitions
        current_group = None
        group_counter = 0
        row_in_group = 0
        
        for idx, row in df_temp.iterrows():
            gk = row['_group_key']
            n_rows = group_sizes[gk]
            
            if gk != current_group:
                # New group starts
                current_group = gk
                group_counter += 1
                row_in_group = 0
                group_id = f"vgrp_{group_counter}"
            else:
                row_in_group += 1
            
            # --- Determine row class ---
            if n_rows <= 1:
                # Single-exposure visit: render normally, no grouping needed
                html += "                    <tr>\n"
            elif row_in_group == 0:
                # First exposure = summary header row
                html += f'                    <tr class="group-header-row" data-group="{group_id}">\n'
            else:
                # Subsequent exposures = hidden child rows
                html += f'                    <tr class="group-child-row" data-group="{group_id}">\n'
            
            # --- Render cells ---
            for col_idx, col in enumerate(df.columns):
                value = row[col]
                cell_content = _format_cell(col, value)
                
                # Prepend expand button to the first column of multi-exposure header row
                if col_idx == 0 and n_rows > 1 and row_in_group == 0:
                    cell_content = (
                        f'<span class="expand-btn" '
                        f'onclick="toggleVisitGroup(\'{group_id}\', this); event.stopPropagation();" '
                        f'title="Click to show/hide {n_rows} exposures">▶</span>'
                        f'{cell_content}'
                        f'<span class="exp-count-badge">{n_rows} exp</span>'
                    )
                
                html += f"                        <td>{cell_content}</td>\n"
            
            html += "                    </tr>\n"
            
    else:
        # Fallback: grouping columns not available, render all rows flat
        for idx, row in df.iterrows():
            html += "                    <tr>\n"
            for col in df.columns:
                cell_content = _format_cell(col, row[col])
                html += f"                        <td>{cell_content}</td>\n"
            html += "                    </tr>\n"
    
    html += f"""                </tbody>
            </table>
        </div>
    </div>
    
    <div class="stats">
        <div class="stat-item">
            <div class="stat-label">Total Rows</div>
            <div class="stat-value">{total_rows}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Total Columns</div>
            <div class="stat-value">{total_cols}</div>
        </div>
    </div>
    
    <!-- Modal for displaying visit file content -->
    <div id="visitModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 id="modalTitle">Visit File Content</h2>
                <div class="modal-controls">
                    <div class="syntax-toggle">
                        <label for="syntaxHighlight">Syntax Highlighting:</label>
                        <input type="checkbox" id="syntaxHighlight" checked onchange="toggleSyntaxHighlighting()">
                    </div>
                    <span class="close" onclick="closeModal()">&times;</span>
                </div>
            </div>
            <div class="modal-body">
                <div id="scrollHint" class="scroll-hint" style="display: none;">
                    ⟷ Scroll horizontally and vertically to view all content
                </div>
                <pre id="visitContentDisplay" class="visit-content"></pre>
            </div>
            <div class="modal-footer">
                <div class="footer-info">
                    <span id="fileSize"></span>
                </div>
                <div class="footer-buttons">
                    <button class="copy-btn" onclick="copyToClipboard()">Copy to Clipboard</button>
                    <button class="download-btn" onclick="downloadVisitFile()">Download</button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Generated by OPUP Parser | Visit files are contained in the .tgz archive</p>
        <p>Tip: Use Shift + Mouse Wheel to scroll horizontally | Click visit file names to view content</p>
    </div>
    
    <script>
        // Store visit file contents - loaded from JSON
        const visitContentsRaw = {visit_contents_raw_json};
        const visitContentsHighlighted = {visit_contents_highlighted_json};
        
        console.log('Loaded visit contents:', Object.keys(visitContentsRaw).length, 'files');
        
        let currentVisitFile = '';
        let syntaxEnabled = true;
        
        function showVisitContent(filename) {{
            console.log('showVisitContent called with:', filename);
            
            const modal = document.getElementById('visitModal');
            const titleElement = document.getElementById('modalTitle');
            const contentElement = document.getElementById('visitContentDisplay');
            const fileSizeElement = document.getElementById('fileSize');
            const scrollHint = document.getElementById('scrollHint');
            const syntaxCheckbox = document.getElementById('syntaxHighlight');
            
            if (!modal || !titleElement || !contentElement) {{
                console.error('Modal elements not found!');
                return;
            }}
            
            currentVisitFile = filename;
            titleElement.textContent = 'Visit File: ' + filename;
            
            if (visitContentsRaw[filename]) {{
                console.log('Found content for:', filename, 'Length:', visitContentsRaw[filename].length);
                
                // Display with or without syntax highlighting
                if (syntaxEnabled && visitContentsHighlighted[filename]) {{
                    contentElement.innerHTML = visitContentsHighlighted[filename];
                }} else {{
                    contentElement.textContent = visitContentsRaw[filename];
                }}
                
                // Show file size info
                const sizeKB = (visitContentsRaw[filename].length / 1024).toFixed(2);
                const lines = visitContentsRaw[filename].split('\\n').length;
                fileSizeElement.textContent = `Size: ${{sizeKB}} KB | Lines: ${{lines}}`;
                
                // Show scroll hint briefly
                scrollHint.style.display = 'block';
                setTimeout(() => {{
                    scrollHint.style.display = 'none';
                }}, 4000);
            }} else {{
                console.warn('No content found for:', filename);
                console.log('Available files:', Object.keys(visitContentsRaw));
                contentElement.textContent = 'Visit file content not available.\\n\\nAvailable files:\\n' + 
                    Object.keys(visitContentsRaw).join('\\n');
                fileSizeElement.textContent = '';
            }}
            
            modal.style.display = 'block';
            console.log('Modal displayed');
        }}
        
        function toggleSyntaxHighlighting() {{
            const checkbox = document.getElementById('syntaxHighlight');
            syntaxEnabled = checkbox.checked;
            
            // Refresh the current display if a file is open
            if (currentVisitFile) {{
                const contentElement = document.getElementById('visitContentDisplay');
                if (syntaxEnabled && visitContentsHighlighted[currentVisitFile]) {{
                    contentElement.innerHTML = visitContentsHighlighted[currentVisitFile];
                }} else {{
                    contentElement.textContent = visitContentsRaw[currentVisitFile];
                }}
            }}
        }}
        
        function closeModal() {{
            const modal = document.getElementById('visitModal');
            modal.style.display = 'none';
            console.log('Modal closed');
        }}
        
        function downloadVisitFile() {{
            if (!currentVisitFile || !visitContentsRaw[currentVisitFile]) {{
                alert('No visit file content available to download.');
                return;
            }}
            
            const content = visitContentsRaw[currentVisitFile];
            
            // Create blob and download
            const blob = new Blob([content], {{ type: 'text/plain' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = currentVisitFile;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            console.log('Downloaded:', currentVisitFile);
        }}
        
        function copyToClipboard() {{
            if (!currentVisitFile || !visitContentsRaw[currentVisitFile]) {{
                alert('No visit file content available to copy.');
                return;
            }}
            
            const content = visitContentsRaw[currentVisitFile];
            
            // Use the Clipboard API
            navigator.clipboard.writeText(content).then(() => {{
                // Temporarily change button text
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = 'Copied!';
                btn.style.backgroundColor = '#27ae60';
                
                setTimeout(() => {{
                    btn.textContent = originalText;
                    btn.style.backgroundColor = '#3498db';
                }}, 2000);
                
                console.log('Copied to clipboard:', currentVisitFile);
            }}).catch(err => {{
                console.error('Failed to copy:', err);
                alert('Failed to copy to clipboard');
            }});
        }}
        
        // Close modal when clicking outside of it
        window.onclick = function(event) {{
            const modal = document.getElementById('visitModal');
            if (event.target == modal) {{
                closeModal();
            }}
        }}
        
        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});
        
        // Test that everything loaded
        console.log('Script loaded successfully');

        function toggleVisitGroup(groupId, btn) {{
            var childRows = document.querySelectorAll('tr.group-child-row[data-group="' + groupId + '"]');
            var isExpanded = btn.classList.contains('expanded');
            
            for (var i = 0; i < childRows.length; i++) {{
                if (isExpanded) {{
                    childRows[i].classList.remove('visible');
                }} else {{
                    childRows[i].classList.add('visible');
                }}
            }}
            
            if (isExpanded) {{
                btn.classList.remove('expanded');
                btn.textContent = '▶';
                btn.title = btn.title.replace('hide', 'show');
            }} else {{
                btn.classList.add('expanded');
                btn.textContent = '▼';
                btn.title = btn.title.replace('show', 'hide');
            }}
        }}
        
        function toggleAllVisitGroups(expand) {{
            var buttons = document.querySelectorAll('.expand-btn');
            for (var i = 0; i < buttons.length; i++) {{
                var btn = buttons[i];
                var groupId = btn.closest('tr').getAttribute('data-group');
                var isExpanded = btn.classList.contains('expanded');
                if (expand && !isExpanded) {{
                    toggleVisitGroup(groupId, btn);
                }} else if (!expand && isExpanded) {{
                    toggleVisitGroup(groupId, btn);
                }}
            }}
        }}

    </script>
</body>
</html>
"""
    
    return html
import re

def syntax_highlight_visit_content(content):
    """
    Apply syntax highlighting to STOL visit file content.
    Highlights groups, sequences, activities, commands, parameters, etc.
    
    Args:
        content: Raw visit file content string
    
    Returns:
        HTML string with syntax highlighting
    """
    if not content or content == "Visit file content not available":
        return content
    
    lines = content.split('\n')
    highlighted_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            highlighted_lines.append('')
            continue
        
        original_line = line
        highlighted_line = line
        
        # 1. Comment lines (starting with ;@)
        if line.strip().startswith(';@'):
            highlighted_line = f'<span class="stol-comment">{html_module.escape(line)}</span>'
        
        # 2. VISIT line
        elif line.strip().startswith('VISIT,'):
            # Parse: VISIT, V0010801001001001002, EARLY=..., LATE=..., CUTOFF=..., CONVST=...;
            match = re.match(r'^(VISIT,\s*)([^,]+)(,\s*.+)$', line)
            if match:
                keyword, visit_id, rest = match.groups()
                # Parse the rest for key=value pairs
                rest_parts = []
                current = ""
                for char in rest:
                    if char == ',' or char == ';':
                        if '=' in current:
                            key, val = current.split('=', 1)
                            rest_parts.append(f'<span class="stol-param-name">{html_module.escape(key)}</span>=<span class="stol-string">{html_module.escape(val)}</span>')
                        else:
                            rest_parts.append(html_module.escape(current))
                        rest_parts.append(char)
                        current = ""
                    else:
                        current += char
                if current:
                    if '=' in current:
                        key, val = current.split('=', 1)
                        rest_parts.append(f'<span class="stol-param-name">{html_module.escape(key)}</span>=<span class="stol-string">{html_module.escape(val)}</span>')
                    else:
                        rest_parts.append(html_module.escape(current))
                
                rest_highlighted = ''.join(rest_parts)
                highlighted_line = f'<span class="stol-keyword">{html_module.escape(keyword)}</span><span class="stol-constant">{html_module.escape(visit_id)}</span>{rest_highlighted}'
            else:
                highlighted_line = f'<span class="stol-keyword">{html_module.escape(line)}</span>'
        
        # 3. GROUP line
        elif re.match(r'^\s*GROUP,', line):
            # Parse: GROUP, 01, CONGRP=NONE;
            match = re.match(r'^(\s*)(GROUP)(,\s*)(\d+)(,\s*)([A-Z_]+)(=)([^;]+)(;?)(.*)$', line)
            if match:
                indent, keyword, comma1, number, comma2, param_name, equals, param_value, semicolon, comment = match.groups()
                highlighted_line = (f'{indent}<span class="stol-keyword">{keyword}</span>{comma1}'
                                  f'<span class="stol-number">{number}</span>{comma2}'
                                  f'<span class="stol-param-name">{param_name}</span>{equals}'
                                  f'<span class="stol-constant">{param_value}</span>'
                                  f'<span class="stol-punctuation">{semicolon}</span>{comment}')
            else:
                highlighted_line = f'<span class="stol-keyword">{html_module.escape(line)}</span>'
        
        # 4. SEQ line
        elif re.match(r'^\s*SEQ,', line):
            # Parse: SEQ, 1, CONSEQ=NONE;
            match = re.match(r'^(\s*)(SEQ)(,\s*)(\d+)(,\s*)([A-Z_]+)(=)([^;]+)(;?)(.*)$', line)
            if match:
                indent, keyword, comma1, number, comma2, param_name, equals, param_value, semicolon, comment = match.groups()
                highlighted_line = (f'{indent}<span class="stol-keyword">{keyword}</span>{comma1}'
                                  f'<span class="stol-number">{number}</span>{comma2}'
                                  f'<span class="stol-param-name">{param_name}</span>{equals}'
                                  f'<span class="stol-constant">{param_value}</span>'
                                  f'<span class="stol-punctuation">{semicolon}</span>{comment}')
            else:
                highlighted_line = f'<span class="stol-keyword">{html_module.escape(line)}</span>'
        
        # 5. ACT line with function call
        elif re.match(r'^\s*ACT,', line):
            # Parse: ACT, 01, FUNCTION_NAME(...); PARAM1,PARAM2,...
            match = re.match(r'^(\s*)(ACT)(,\s*)(\d+)(,\s*)([A-Z_]+)(\([^)]*\))(;?)(.*)$', line)
            if match:
                indent, keyword, comma1, number, comma2, func_name, params_with_parens, semicolon, comment = match.groups()
                
                # Extract parameters from parentheses
                params_match = re.match(r'\(([^)]*)\)', params_with_parens)
                if params_match:
                    params_str = params_match.group(1)
                    
                    # Highlight individual parameters
                    if params_str.strip():
                        param_parts = []
                        # Split by comma, but be careful with quoted strings
                        current_param = ""
                        in_quotes = False
                        for char in params_str:
                            if char == '"' and (not current_param or current_param[-1] != '\\'):
                                in_quotes = not in_quotes
                                current_param += char
                            elif char == ',' and not in_quotes:
                                param_parts.append(current_param.strip())
                                current_param = ""
                            else:
                                current_param += char
                        if current_param:
                            param_parts.append(current_param.strip())
                        
                        # Highlight each parameter
                        highlighted_params = []
                        for param in param_parts:
                            param = param.strip()
                            # Check if it's a number
                            if re.match(r'^-?\d+\.?\d*([eE][+-]?\d+)?$', param):
                                highlighted_params.append(f'<span class="stol-number">{html_module.escape(param)}</span>')
                            # Check if it's a quoted string
                            elif param.startswith('"') and param.endswith('"'):
                                highlighted_params.append(f'<span class="stol-string">{html_module.escape(param)}</span>')
                            # Check if it's a constant/keyword
                            elif param.isupper() or param in ['TRUE', 'FALSE']:
                                highlighted_params.append(f'<span class="stol-constant">{html_module.escape(param)}</span>')
                            else:
                                highlighted_params.append(f'<span class="stol-param">{html_module.escape(param)}</span>')
                        
                        params_highlighted = '(' + ','.join(highlighted_params) + ')'
                    else:
                        params_highlighted = '()'
                else:
                    params_highlighted = html_module.escape(params_with_parens)
                
                # Highlight comment (parameter names)
                if comment.strip():
                    comment_highlighted = f'<span class="stol-comment">{html_module.escape(comment)}</span>'
                else:
                    comment_highlighted = comment
                
                highlighted_line = (f'{indent}<span class="stol-keyword">{keyword}</span>{comma1}'
                                  f'<span class="stol-number">{number}</span>{comma2}'
                                  f'<span class="stol-function">{func_name}</span>{params_highlighted}'
                                  f'<span class="stol-punctuation">{semicolon}</span>{comment_highlighted}')
            else:
                # Try simpler ACT format without parameters
                match2 = re.match(r'^(\s*)(ACT)(,\s*)(\d+)(,\s*)([A-Z_]+)(;?)(.*)$', line)
                if match2:
                    indent, keyword, comma1, number, comma2, func_name, semicolon, comment = match2.groups()
                    if comment.strip():
                        comment_highlighted = f'<span class="stol-comment">{html_module.escape(comment)}</span>'
                    else:
                        comment_highlighted = comment
                    highlighted_line = (f'{indent}<span class="stol-keyword">{keyword}</span>{comma1}'
                                      f'<span class="stol-number">{number}</span>{comma2}'
                                      f'<span class="stol-function">{func_name}</span>'
                                      f'<span class="stol-punctuation">{semicolon}</span>{comment_highlighted}')
                else:
                    highlighted_line = html_module.escape(line)
        
        # 6. Standalone comment/empty comment line (just semicolon)
        elif line.strip() == ';':
            highlighted_line = f'<span class="stol-comment">{line}</span>'
        
        else:
            # Leave other lines as-is (escape HTML)
            highlighted_line = html_module.escape(line)
        
        highlighted_lines.append(highlighted_line)
    
    return '\n'.join(highlighted_lines)

def write_to_HTML(df, output_html, opup_filepath, keep_GW=True, 
                  sky_plotter_html=None, visit_png_map=None):
    """
    Write DataFrame to HTML file with optional GW column removal.
    
    Args:
        df: DataFrame to export
        output_html: Path to output HTML file
        opup_filepath: Path to OPUP archive
        keep_GW: Whether to keep Guide Window columns
    """

    if visit_png_map is None:
        visit_png_map = {}

    if not keep_GW:
        gw_cols = get_current_gw_columns(df)
        df = df.drop(columns=gw_cols)
    
    html_content = generate_html_report(df, opup_filepath)
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)


# Modified process functions to add HTML export option
def process_OPUPs_html(opup_filepaths, output_dir=None, keep_GW=True):
    """
    Process OPUP files and generate HTML reports.
    """
    for opup_filepath in opup_filepaths:
        if output_dir is None:
            output_dir = Path(opup_filepath).parent.as_posix()
            if not Path(output_dir).is_dir():
                output_dir = find_nontgz_parent(output_dir)
        
        output_html = Path(output_dir).joinpath(Path(opup_filepath).stem + '_report.html').as_posix()
        
        # Parse the opup
        opup_info = parse_OPUP(opup_filepath)
        
        # Move priority columns to the left
        opup_info = prioritize_columns(opup_info, PRIORITY_COLUMNS)
        
        # Write to HTML
        if len(opup_info.columns) > 0:
            write_to_HTML(opup_info, output_html, opup_filepath, keep_GW=keep_GW)
        else:
            print(f'No columns were returned for {opup_filepath}')

def export_unique_visits_for_plotter(df, output_csv):
    """
    Export one row per unique visit for use with roman_plotter.py
    Includes only the columns needed for sky plotting: RA, DEC, Visit_ID, etc.
    
    Args:
        df: Full DataFrame with all exposures
        output_csv: Path to output CSV file (string or Path object)
    """
    # Convert Path to string if necessary
    output_csv = str(output_csv)
    
    if 'Visit_ID' not in df.columns:
        print("Warning: No Visit_ID column found, cannot create unique visits CSV")
        return
    
    # Get unique visits only
    unique_visits_df = df.drop_duplicates(subset=['Visit_ID']).copy()
    
    # Select columns relevant for sky plotting
    columns_to_keep = []
    possible_columns = [
        'Visit_ID', 'RA', 'DEC', 'Position_Angle', 
        'Program_Number', 'Pass_Number', 'Segment_Number', 
        'Observation_Number', 'Visit_Number',
        'Start', 'Duration', 'Science_Instrument', 
        'WFI_Optical_Element', 'Visit_File_Name',
        'Earliest_Start_Time', 'Latest_Start_Time',
        'Intended_Purpose'  # Add this for color-coding in the plotter
    ]
    
    for col in possible_columns:
        if col in unique_visits_df.columns:
            columns_to_keep.append(col)
    
    # Create the filtered dataframe
    plotter_df = unique_visits_df[columns_to_keep].copy()
    
    # Save to CSV
    plotter_df.to_csv(output_csv, index=False)
    print(f"  📊 Generated sky plotter CSV: {output_csv} ({len(plotter_df)} unique visits)")
    
    return output_csv

def _parse_sun_date(opup_info):
    """Extract observation date from the first visit's Start time for Sun position."""
    from datetime import datetime, timezone, timedelta
    
    if 'Start' not in opup_info.columns:
        print(f"  ⚠️  No 'Start' column; using today for Sun position")
        return datetime.now(timezone.utc)
    
    first_start = opup_info['Start'].dropna().iloc[0] if len(opup_info['Start'].dropna()) > 0 else None
    if first_start is None:
        print(f"  ⚠️  No valid Start times; using today for Sun position")
        return datetime.now(timezone.utc)
    
    # Try YYYY-DDD format (day of year), e.g. "2026-276-13:00:51 TAI"
    doy_match = re.match(
        r'^(\d{4})-(\d{1,3})(?:-(\d{2}:\d{2}:\d{2}))?\s*(?:TAI|UTC|TDB|TT)?$',
        str(first_start).strip(), re.I
    )
    if doy_match:
        year = int(doy_match.group(1))
        doy = int(doy_match.group(2))
        if 1 <= doy <= 366:
            sun_date = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
            print(f"  ☀️  Using date from first visit: {sun_date.strftime('%Y-%m-%d')} (DOY {doy})")
            return sun_date
    
    # Try standard date formats
    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
        try:
            sun_date = datetime.strptime(str(first_start).split()[0], fmt).replace(tzinfo=timezone.utc)
            print(f"  ☀️  Using date from first visit: {sun_date.strftime('%Y-%m-%d')}")
            return sun_date
        except ValueError:
            continue
    
    print(f"  ⚠️  Could not parse visit date, using today: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    return datetime.now(timezone.utc)

def _generate_sky_plotter(opup_stem, output_dir, plotter_csv, sun_date):
    """Generate the interactive sky plotter HTML via roman_plotter."""
    sky_plotter_html = output_dir / f"{opup_stem}_skymap.html"

    try:
        from roman_opup_tools import roman_plotter

        plotter_data = pd.read_csv(plotter_csv)
        data_json = plotter_data.to_json(orient='records')
        preloaded_datasets = [{
            'fileName': plotter_csv.name,
            'data_json': data_json
        }]

        sun_position = roman_plotter.get_sun_position(sun_date)
        print(f"  ☀️  Sun RA={sun_position['ra']:.2f}°, Dec={sun_position['dec']:.2f}° "
              f"(Galactic: l={sun_position['l']:.2f}°, b={sun_position['b']:.2f}°)")

        html_content = roman_plotter.generate_html(
            preloaded_datasets=preloaded_datasets,
            sun_position=sun_position
        )

        with open(sky_plotter_html, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"  ✅ Generated sky plotter: {sky_plotter_html}")
        return sky_plotter_html

    except Exception as e:
        print(f"  ⚠️  Could not generate sky plotter: {e}")
        import traceback
        traceback.print_exc()
        return None
    

from roman_opup_tools.roman_attitude import RomanPointing
from astropy.time import Time

def get_pitch_and_roll(ra, dec, v3pa, obs_time):
    """
    Given the actual pointing from a visit file, compute the pitch
    and off-nominal roll angle.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees (V1 boresight)
    dec : float
        Declination in degrees (V1 boresight)
    v3pa : float
        V3 Position Angle in degrees (from visit file)
    obs_time : str or datetime or astropy.Time
        Observation time (needed for Sun position)

    Returns
    -------
    pitch : float
        Pitch angle in degrees. 0° = Sun perpendicular to boresight.
        Positive = pitched away from Sun.
    roll : float
        Off-nominal roll in degrees. 0° = nominal (Z toward Sun).
        This is the difference between the actual V3PA and the 
        nominal V3PA that the spacecraft would have at roll=0.
    sun_angle : float
        Sun–target separation in degrees.
    nominal_v3pa : float
        The V3PA the spacecraft would have at nominal roll=0.
    """
    # 1. Create RomanPointing at the observation time
    rp = RomanPointing(observation_date=Time(obs_time))

    # 2. Point at the target with nominal roll (roll=0)
    #    This builds the attitude matrix with Z toward the Sun
    rp.set_target_using_radec(ra, dec, roll=0.0)

    # 3. Pitch is purely a function of Sun–target geometry
    #    (independent of roll — it only depends on RA, Dec, and date)
    pitch = rp.get_pitch_angle()          # returns Quantity in degrees
    sun_angle = rp.get_sun_angle()        # Sun-target separation in degrees

    # 4. Get the nominal V3PA (what PA would be at roll=0)
    nominal_v3pa = rp.get_position_angle()  # returns Quantity in degrees

    # 5. Roll = difference between actual V3PA and nominal V3PA
    #    Wrap to [-180, +180]
    roll = (v3pa - nominal_v3pa.value + 180) % 360 - 180

    return {
        'pitch': pitch.value,
        'roll': roll,
        'sun_angle': sun_angle,
        'nominal_v3pa': nominal_v3pa.value,
        'actual_v3pa': v3pa
    }

def add_attitude_columns(df):
    """
    Add Sun_Angle, Pitch, and Off-Nominal_Roll columns to the DataFrame.
    """
    required = ['RA', 'DEC', 'Position_Angle', 'Start']
    if not all(col in df.columns for col in required):
        missing = [c for c in required if c not in df.columns]
        print(f"  ⚠️  Cannot compute attitude columns — missing: {missing}")
        return df
    
    try:
        from roman_opup_tools.roman_attitude import RomanPointing, OEMEphemeris, get_sun_from_l2_jpl, get_sun_from_rst
        from astropy.time import Time
        from astropy.coordinates import SkyCoord
        from astropy import units as u
    except ImportError:
        print("  ⚠️  roman_attitude not available — skipping attitude columns")
        return df

    sun_angles = []
    pitches = []
    rolls = []
    oem = OEMEphemeris("RST_EPH_PRED_LONG_2026250_2027065_01.oem")


    for idx, row in df.iterrows():
        if any(pd.isna(row.get(c)) for c in required):
            sun_angles.append(None)
            pitches.append(None)
            rolls.append(None)
            continue

        try:
            ra = float(row['RA'])
            dec = float(row['DEC'])
            v3pa = float(row['Position_Angle'])
            start_str = str(row['Start']).strip()

            obs_time = _parse_obs_time(start_str)
            if obs_time is None:
                sun_angles.append(None)
                pitches.append(None)
                rolls.append(None)
                continue

            # ── Sun position from JPL for this exact time ──
            # sun_ra, sun_dec = get_sun_from_l2_jpl(obs_time)
            sun_ra, sun_dec = get_sun_from_rst(obs_time, oem)
            sun_coord = SkyCoord(ra=sun_ra*u.deg, dec=sun_dec*u.deg, frame='icrs')

            # ── Sun angle: pure geometry ──
            target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            sun_angle = sun_coord.separation(target).deg

            # ── Pitch ──
            pitch_val = sun_angle - 90.0

            # ── Roll: fresh RomanPointing for this exact time ──
            rp = RomanPointing(observation_date=obs_time)
            rp.set_target_using_radec(ra, dec, roll=0.0)
            nominal_v3pa = rp.get_position_angle()
            roll_val = (v3pa - nominal_v3pa.value + 180) % 360 - 180

            sun_angles.append(round(sun_angle, 2))
            pitches.append(round(pitch_val, 2))
            rolls.append(round(roll_val, 2))

        except Exception as e:
            sun_angles.append(None)
            pitches.append(None)
            rolls.append(None)

    df['Sun_Angle [calc]'] = sun_angles
    df['Pitch [calc]'] = pitches
    df['Off-Normal_Roll [calc]'] = rolls

    n_computed = sum(1 for v in pitches if v is not None)
    n_nonzero_pitch = sum(1 for v in pitches if v is not None and abs(v) > 0.1)
    print(f"  🧭 Computed attitude for {n_computed}/{len(df)} rows ({n_nonzero_pitch} with |pitch| > 0.1°)")

    valid_pitches = [v for v in pitches if v is not None]
    if valid_pitches:
        print(f"  📐 Pitch range: [{min(valid_pitches):.2f}°, {max(valid_pitches):.2f}°]")
    valid_sun = [v for v in sun_angles if v is not None]
    if valid_sun:
        print(f"  ☀️  Sun angle range: [{min(valid_sun):.2f}°, {max(valid_sun):.2f}°]")

    return df


def _parse_obs_time(start_str):
    """
    Parse a Start time string from the visit file into an astropy Time object.
    Handles formats like '2026-276-13:00:51 TAI' and '2026-10-03T13:00:51'.
    
    Args:
        start_str: str, time string from the Start column
    
    Returns:
        astropy.time.Time or None
    """
    from astropy.time import Time
    from datetime import datetime, timezone, timedelta

    # Strip time scale suffix
    clean = re.sub(r'\s*(TAI|UTC|TDB|TT)\s*$', '', start_str, flags=re.I).strip()

    # Try YYYY-DDD-HH:MM:SS (day of year)
    doy_match = re.match(r'^(\d{4})-(\d{1,3})(?:-(\d{2}:\d{2}:\d{2}))?$', clean)
    if doy_match:
        year = int(doy_match.group(1))
        doy = int(doy_match.group(2))
        time_part = doy_match.group(3) or '00:00:00'
        if 1 <= doy <= 366:
            dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
            h, m, s = [int(x) for x in time_part.split(':')]
            dt = dt.replace(hour=h, minute=m, second=s)
            return Time(dt)

    # Try standard ISO formats
    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
        try:
            dt = datetime.strptime(clean, fmt).replace(tzinfo=timezone.utc)
            return Time(dt)
        except ValueError:
            continue

    return None

def generate_integrated_report(opup_filepath, output_dir=None, keep_GW=True, generate_pngs=False):
    """
    Generate both the detailed OPUP HTML report and the sky plotter visualization.
    
    Args:
        opup_filepath: Path to OPUP .tgz archive
        output_dir: Output directory (defaults to same as OPUP)
        keep_GW: Whether to keep Guide Window columns
        generate_pngs: Whether to generate sky plot PNGs via roman_visit_viewer
                       (default: False, uses Aladin Lite embedded viewer instead)
    
    Returns:
        Tuple of (html_report_path, sky_plotter_path, csv_path, archive_path)
    """
    from pathlib import Path
    from datetime import datetime, timezone
    
    # ── Setup ──
    if output_dir is None:
        output_dir = Path(opup_filepath).parent
        if not output_dir.is_dir():
            output_dir = find_nontgz_parent(opup_filepath)
    
    output_dir = Path(output_dir)
    opup_stem = Path(opup_filepath).stem.replace('.tgz', '').replace('.tar', '')
    
    generated_files = []
    step = 0
    
    print(f"\n{'='*60}")
    print(f"Generating Integrated OPUP Report")
    print(f"{'='*60}\n")
    
    # ── Step 1: Parse OPUP ──
    step += 1
    print(f"Step {step}: Parsing OPUP...")
    opup_info = parse_OPUP(opup_filepath)
    
    # ── Step 2: Add attitude columns ──
    step += 1
    print(f"\nStep {step}: Computing spacecraft attitude (pitch & roll)...")
    opup_info = add_attitude_columns(opup_info)
    opup_info = prioritize_columns(opup_info, PRIORITY_COLUMNS)

    # ── Step 3: Compute V1 and WFI_CEN pointing from quaternion ──
    step += 1
    print(f"\nStep {step}: Computing V1 and WFI_CEN pointing from quaternions...")
    opup_info = add_pointing_columns(opup_info)

    opup_info = prioritize_columns(opup_info, PRIORITY_COLUMNS)


    # ── Step 3: Precompute WFI footprints from quaternions ──
    step += 1
    print(f"\nStep {step}: Precomputing WFI footprints from pointing quaternions...")
    wfi_footprints = precompute_wfi_footprints(opup_info)

    # ── Step 4: Extract date for Sun position ──
    sun_date = _parse_sun_date(opup_info)
    
    # ── Step 5: Sky plotter CSV ──
    step += 1
    print(f"\nStep {step}: Creating sky plotter CSV...")
    plotter_csv = output_dir / f"{opup_stem}_skymap.csv"
    export_unique_visits_for_plotter(opup_info, plotter_csv)
    generated_files.append(plotter_csv)
    
    # ── Step 6: Interactive sky plotter HTML (roman_plotter) ──
    step += 1
    print(f"\nStep {step}: Generating interactive sky plotter...")
    sky_plotter_html = _generate_sky_plotter(opup_stem, output_dir, plotter_csv, sun_date)
    if sky_plotter_html:
        generated_files.append(sky_plotter_html)
    
    # ── Step 7: Sky plot PNGs (optional, off by default) ──
    visit_png_map = {}
    skyplot_mosaic_filename = ''
    
    if generate_pngs:
        step += 1
        print(f"\nStep {step}: Generating sky plot PNGs via roman_visit_viewer...")
        try:
            visit_png_map = generate_sky_plot_pngs(opup_filepath, output_dir, opup_info)
            print(f"   Generated {len(visit_png_map)} sky plot PNGs")
        except Exception as e:
            print(f"   ⚠️  Sky plot generation failed: {e}")
            visit_png_map = {}

        # ── Step 7b: Sky plot mosaic HTML ──
        if visit_png_map:
            step += 1
            print(f"\nStep {step}: Generating sky plot mosaic page...")
            mosaic_path = output_dir / f"{opup_stem}_skyplots.html"
            result = generate_skyplot_mosaic_html(
                visit_png_map, opup_stem, mosaic_path, df=opup_info
            )
            if result:
                skyplot_mosaic_filename = mosaic_path.name
                generated_files.append(mosaic_path)
    else:
        print("\n  ⏭️  Skipping sky plot PNGs (using embedded Aladin Lite viewer)")

    # ── Step 8: Main HTML report ──
    step += 1
    print(f"\nStep {step}: Generating detailed HTML report...")
    html_report = output_dir / f"{opup_stem}_report.html"
    html_content = generate_html_report(
        opup_info, opup_filepath, sky_plotter_html,
        visit_png_map=visit_png_map,
        skyplot_mosaic_filename=skyplot_mosaic_filename
    )

    # ── Step 8b: Embed Aladin Lite sky viewer ──
    if wfi_footprints:
        print(f"  🌐 Embedding Aladin Lite sky viewer...")
        html_content = inject_aladin_into_html(html_content, wfi_footprints, df=opup_info)
        print(f"  ✅ Aladin viewer embedded ({len(wfi_footprints)} footprints)")

    with open(html_report, 'w', encoding='utf-8') as f:
        f.write(html_content)
    generated_files.append(html_report)
    
    # ── Step 9: Full CSV ──
    step += 1
    print(f"\nStep {step}: Generating full CSV...")
    full_csv = output_dir / f"{opup_stem}_full.csv"
    write_to_CSV(opup_info, full_csv, keep_GW=keep_GW)
    generated_files.append(full_csv)

    # ── Step 10: Package everything ──
    step += 1
    print(f"\nStep {step}: Packaging report archive...")
    archive_path = package_report_archive(opup_stem, output_dir, generated_files)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"✅ Complete! ({step} steps)")
    print(f"{'='*60}")
    print(f"\n📄 Detailed Report: {html_report}")
    if sky_plotter_html:
        print(f"🌌 Sky Plotter:     {sky_plotter_html}")
    if skyplot_mosaic_filename:
        print(f"🔭 Sky Plot Mosaic: {output_dir / skyplot_mosaic_filename}")
    print(f"📊 Full CSV:        {full_csv}")
    print(f"📊 Sky Plot CSV:    {plotter_csv}")
    if archive_path:
        print(f"📦 Report Archive:  {archive_path}")
    print(f"\nOpen {html_report} in your browser to get started!\n")
    
    return html_report, sky_plotter_html, full_csv, archive_path

# Add HTML option to command-line parser
def setup_parser_with_html():
    parser = argparse.ArgumentParser(description='Parse OPUP files.')
    parser.add_argument('-opup', '--opup_filepath', type=str, nargs='+', help='Path(s) to the OPUP file(s)', default=[])
    parser.add_argument('-opup_dir', '--opup_directory', type=str, help='Directory containing OPUP .tgz archives (will process all found)', default=None)
    parser.add_argument('-scf', '--scf_filepath', type=str, nargs='+', help='Path(s) to the SCF file(s)', default=[])
    parser.add_argument('-visit', '--visit_filepath', type=str, nargs='+', help='Path(s) to the visit file(s)', default=[])
    parser.add_argument('-odir', '--output_dir', type=str, help='Output file directory', default=None)
    parser.add_argument('--keep_GW', action='store_true', help='Keep Guide Window information in the output.')
    parser.add_argument('--format', type=str, choices=['csv', 'html', 'both'], default='html', 
                       help='Output format: csv, html, or both')
    return parser


def setup_parser():
    parser = argparse.ArgumentParser(description='Parse OPUP files.')
    parser.add_argument('-opup', '--opup_filepath', type=str, nargs='+', help='Path(s) to the OPUP file(s)', default=[])
    parser.add_argument('-opup_dir', '--opup_directory', type=str, help='Directory containing OPUP .tgz archives (will process all found)', default=None)
    parser.add_argument('-scf', '--scf_filepath', type=str, nargs='+', help='Path(s) to the SCF file(s)', default=[])
    parser.add_argument('-visit', '--visit_filepath', type=str, nargs='+', help='Path(s) to the visit file(s)', default=[])
    parser.add_argument('-odir', '--output_dir', type=str, help='Output file directory', default=None)
    parser.add_argument('--keep_GW', action='store_true', help='Keep Guide Window information in the output.')
    parser.add_argument('--pngs', action='store_true', default=False,
                       help='Generate sky plot PNGs via roman_visit_viewer (slower, off by default)')
    parser.add_argument('--gantt', type=str, help='Generate Gantt chart from aggregated CSV file')
    parser.add_argument('--format', type=str, choices=['csv', 'html', 'both', 'integrated'], default='integrated', 
                       help='Output format: csv, html, both, or integrated (html + sky plot)')
    return parser


def find_opup_files_in_directory(directory):
    """
    Recursively find all .tgz OPUP archive files in the given directory.
    
    Args:
        directory: Path to directory to search
        
    Returns:
        List of paths to .tgz files found
    """
    from pathlib import Path
    
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Warning: Directory does not exist: {directory}")
        return []
    
    if not directory.is_dir():
        print(f"Warning: Path is not a directory: {directory}")
        return []
    
    # Find all .tgz files recursively
    opup_files = list(directory.rglob('*.tgz'))
    
    # Also look for .tar.gz files
    opup_files.extend(directory.rglob('*.tar.gz'))
    
    # Convert to strings
    opup_files = [str(f) for f in opup_files]
    
    print(f"Found {len(opup_files)} OPUP archive(s) in {directory}")
    for opup_file in opup_files:
        print(f"  - {Path(opup_file).name}")
    
    return opup_files

def aggregate_opup_dataframes(opup_filepaths, output_dir=None, keep_GW=True):
    """
    Process multiple OPUP files and aggregate all data into a single DataFrame.
    
    Args:
        opup_filepaths: List of paths to OPUP .tgz archives
        output_dir: Output directory for aggregated results
        keep_GW: Whether to keep Guide Window columns
        
    Returns:
        pd.DataFrame: Aggregated DataFrame containing all OPUP data
    """
    from pathlib import Path
    import pandas as pd
    
    if not opup_filepaths:
        print("Warning: No OPUP files to process")
        return pd.DataFrame()
    
    print(f"\n{'='*60}")
    print(f"Aggregating {len(opup_filepaths)} OPUP file(s)")
    print(f"{'='*60}\n")
    
    all_opup_data = []
    
    for i, opup_filepath in enumerate(opup_filepaths, 1):
        try:
            print(f"[{i}/{len(opup_filepaths)}] Processing: {Path(opup_filepath).name}")
            
            # Parse OPUP
            opup_info = parse_OPUP(opup_filepath)
            opup_info = prioritize_columns(opup_info, PRIORITY_COLUMNS)
            
            if len(opup_info) > 0:
                # Add source file column to track which OPUP each row came from
                opup_info['OPUP_Source'] = Path(opup_filepath).stem
                all_opup_data.append(opup_info)
                print(f"  ✅ Extracted {len(opup_info)} visits")
            else:
                print(f"  ⚠️  No data extracted")
                
        except Exception as e:
            print(f"  ❌ Error processing {Path(opup_filepath).name}: {e}")
            continue
    
    # Combine all dataframes
    if not all_opup_data:
        print("\n❌ No data was successfully extracted from any OPUP files")
        return pd.DataFrame()
    
    print(f"\n{'='*60}")
    print(f"Combining data from {len(all_opup_data)} OPUP file(s)...")
    print(f"{'='*60}\n")
    
    # Concatenate all dataframes
    aggregated_df = pd.concat(all_opup_data, ignore_index=True)
    
    # Sort by OPUP source and visit ID if available
    if 'Visit_ID' in aggregated_df.columns:
        aggregated_df = aggregated_df.sort_values(['OPUP_Source', 'Visit_ID'])
    else:
        aggregated_df = aggregated_df.sort_values('OPUP_Source')
    
    print(f"✅ Total visits aggregated: {len(aggregated_df)}")
    print(f"   Columns: {len(aggregated_df.columns)}")
    print(f"   OPUPs represented: {aggregated_df['OPUP_Source'].nunique()}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(opup_filepaths[0]).parent
        if not output_dir.is_dir():
            output_dir = find_nontgz_parent(str(output_dir))
    
    output_dir = Path(output_dir)
    
    # Save aggregated CSV
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_csv = output_dir / f"aggregated_opups_{timestamp}.csv"
    
    print(f"\n💾 Saving aggregated data to: {output_csv}")
    write_to_CSV(aggregated_df, str(output_csv), keep_GW=keep_GW)
    
    # Generate summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    
    # Count visits per OPUP
    visits_per_opup = aggregated_df.groupby('OPUP_Source').size()
    print("\nVisits per OPUP:")
    for opup_name, count in visits_per_opup.items():
        print(f"  {opup_name}: {count} visits")
    
    # Check for common columns
    if 'Filter' in aggregated_df.columns:
        print("\nFilter distribution:")
        filter_counts = aggregated_df['Filter'].value_counts()
        for filter_name, count in filter_counts.items():
            print(f"  {filter_name}: {count} visits")
    
    if 'Optical_Element' in aggregated_df.columns:
        print("\nOptical Element distribution:")
        oe_counts = aggregated_df['Optical_Element'].value_counts()
        for oe_name, count in oe_counts.items():
            print(f"  {oe_name}: {count} visits")
    
    print(f"\n{'='*60}")
    print(f"✅ Aggregation complete!")
    print(f"{'='*60}\n")
    
    return aggregated_df, str(output_csv)

def create_opup_gantt_chart(csv_filepath, output_dir=None):
    """
    Create a Gantt chart showing schedulability windows for OPUPs.
    
    Shows:
    - Earliest start to Latest start window (schedulability window)
    - Actual scheduled start time
    - Visit execution windows
    
    Args:
        csv_filepath: Path to aggregated CSV file
        output_dir: Output directory for the chart (defaults to same as CSV)
        
    Returns:
        Path to generated HTML Gantt chart
    """
    import pandas as pd
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    from pathlib import Path
    
    print(f"\n{'='*60}")
    print("Creating OPUP Schedulability Gantt Chart")
    print(f"{'='*60}\n")
    
    # Load data
    df = pd.read_csv(csv_filepath)
    
    # Parse time columns
    def parse_time_column(time_str):
        """Parse time string in format '2026-274-14:23:59 TAI' to datetime"""
        if pd.isna(time_str):
            return None
        
        time_str = str(time_str).strip()
        # Remove timezone suffix
        time_str = time_str.split()[0] if ' ' in time_str else time_str
        
        # Parse YYYY-DDD-HH:MM:SS format
        try:
            parts = time_str.split('-')
            year = int(parts[0])
            doy = int(parts[1])
            
            if len(parts) > 2:
                time_parts = parts[2].split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                second = int(time_parts[2]) if len(time_parts) > 2 else 0
            else:
                hour = minute = second = 0
            
            # Convert day-of-year to datetime
            dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second)
            return dt
        except Exception as e:
            print(f"Warning: Could not parse time '{time_str}': {e}")
            return None
    
    # Parse all time columns
    df['Start_dt'] = df['Start'].apply(parse_time_column)
    df['Earliest_Start_dt'] = df['Earliest_Start_Time'].apply(parse_time_column)
    df['Latest_Start_dt'] = df['Latest_Start_Time'].apply(parse_time_column)
    df['Latest_End_dt'] = df['Latest_End_Time'].apply(parse_time_column)
    
    # Get unique OPUPs and sort by earliest Start time
    opup_start_times = {}
    for opup in df['OPUP_Source'].unique():
        opup_df = df[df['OPUP_Source'] == opup]
        min_start = opup_df['Start_dt'].min()
        opup_start_times[opup] = min_start if pd.notna(min_start) else datetime.max
    
    # Sort OPUPs by start time
    opups = sorted(opup_start_times.keys(), key=lambda x: opup_start_times[x])
    
    print(f"Found {len(opups)} unique OPUP(s)")
    print("\nOPUP order (by earliest Start time):")
    for i, opup in enumerate(opups, 1):
        start_time = opup_start_times[opup]
        time_str = start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time != datetime.max else 'N/A'
        print(f"  {i}. {opup.replace('_opup', '').replace('.tgz', '')} @ {time_str}")
    
    # Prepare data for Gantt chart
    gantt_data = []
    colors = {}
    color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
                     '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
    
    # Create OPUP display names with purposes
    opup_display_names = {} 

    # Store OPUP metadata for second pass
    opup_metadata = {}

    for idx, opup in enumerate(opups):
        opup_df = df[df['OPUP_Source'] == opup].copy()
        opup_short = opup.replace('_opup', '').replace('.tgz', '')
        colors[opup] = color_palette[idx % len(color_palette)]
        
        # Check if there's a single unique Intended_Purpose
        if 'Intended_Purpose' in df.columns:
            unique_purposes = opup_df['Intended_Purpose'].dropna().unique()
            if len(unique_purposes) == 1:
                purpose = unique_purposes[0]
                if len(purpose) > 50:
                    purpose = purpose[:47] + "..."
                opup_display_names[opup] = f"{opup_short} | {purpose}"
            else:
                opup_display_names[opup] = opup_short
        else:
            opup_display_names[opup] = opup_short
        
        # Store for second pass
        opup_metadata[opup] = opup_df
        
        print(f"\nProcessing: {opup_display_names[opup]}")
        print(f"  Visits: {len(opup_df)}")
        
        # Get overall OPUP window
        earliest_start = opup_df['Earliest_Start_dt'].min()
        latest_start = opup_df['Latest_Start_dt'].max()
        latest_end = opup_df['Latest_End_dt'].max()
        
        if pd.notna(earliest_start) and pd.notna(latest_start):
            # Add windows first (background)
            gantt_data.append(dict(
                Task=opup_display_names[opup],
                Start=earliest_start,
                Finish=latest_start,
                Resource='Schedulability Window',
                Description=f'OPUP can be scheduled anytime in this window'
            ))
            
    # Second pass: Add scheduled visits on top
    for opup in opups:
        opup_df = opup_metadata[opup]
        for _, visit in opup_df.iterrows():
            if pd.notna(visit['Start_dt']) and 'Duration' in visit and pd.notna(visit['Duration']):
                visit_end = visit['Start_dt'] + timedelta(seconds=visit['Duration'])
                visit_id = visit.get('Visit_ID', 'Unknown')
                
                gantt_data.append(dict(
                    Task=opup_display_names[opup],
                    Start=visit['Start_dt'],
                    Finish=visit_end,
                    Resource='Scheduled Visit',
                    Description=f'Visit {visit_id}'
                ))
        
    if not gantt_data:
        print("❌ No valid scheduling data found")
        return None
    
    # Create figure manually for better control over z-ordering
    from plotly import graph_objects as go

    fig = go.Figure()

    # Create a mapping of OPUP names to y-axis positions
    opup_to_y = {opup_display_names[opup]: idx for idx, opup in enumerate(opups)}

    # Sort gantt_data by Resource type to control drawing order
    # Order: Schedulability Window (bottom), Scheduled Visit (top)
    resource_order = {'Schedulability Window': 0, 'Scheduled Visit': 1}
    gantt_data_sorted = sorted(gantt_data, key=lambda x: resource_order.get(x['Resource'], 2))

    # Color mapping
    color_map = {
        'Schedulability Window': 'rgba(52, 152, 219, 0.3)',  # Light blue
        'Scheduled Visit': 'rgba(46, 204, 113, 0.9)'          # Green (more opaque)
    }

    # Track which resources we've added to legend
    legend_added = set()

    # Plot all bars
    for item in gantt_data_sorted:
        task = item['Task']
        start = item['Start']
        finish = item['Finish']
        resource = item['Resource']
        description = item.get('Description', '')
        
        y_pos = opup_to_y.get(task, 0)
        color = color_map.get(resource, 'gray')
        
        # Determine if this should show in legend
        show_legend = resource not in legend_added
        if show_legend:
            legend_added.add(resource)
        
        # Add the bar as a shape for better control
        fig.add_trace(go.Scatter(
            x=[start, finish, finish, start, start],
            y=[y_pos - 0.4, y_pos - 0.4, y_pos + 0.4, y_pos + 0.4, y_pos - 0.4],
            fill='toself',
            fillcolor=color,
            line=dict(width=1, color='rgba(0,0,0,0.3)' if resource == 'Scheduled Visit' else 'rgba(0,0,0,0.1)'),
            mode='lines',
            name=resource,
            legendgroup=resource,
            showlegend=show_legend,
            hovertemplate=f'<b>{task}</b><br>' +
                        f'{resource}<br>' +
                        f'Start: {start}<br>' +
                        f'End: {finish}<br>' +
                        f'{description}<br>' +
                        '<extra></extra>',
            # Force Scheduled Visits on top with explicit ordering
            legendrank=resource_order.get(resource, 3)
        ))

        # Update layout
        fig.update_layout(
            title='OPUP Schedulability Timeline (Ordered by Start Time)',
            xaxis_title="Date/Time",
            yaxis_title="OPUP (Chronological Order)",
            height=400 + len(opups) * 80,
            font=dict(size=11),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                title="Legend",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01
            ),
            margin=dict(l=300),
            xaxis=dict(
                gridcolor='lightgray',
                showgrid=True,
                tickformat='%Y-%m-%d<br>%H:%M',
                dtick=86400000.0
            ),
            yaxis=dict(
                gridcolor='lightgray',
                showgrid=True,
                tickfont=dict(size=10),
                tickmode='array',
                tickvals=list(range(len(opups))),
                ticktext=[opup_display_names[opup] for opup in opups],
                range=[-0.5, len(opups) - 0.5]
            )
        )    
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date/Time",
            yaxis_title="OPUP (Chronological Order)",
            font=dict(size=11),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                gridcolor='lightgray',
                showgrid=True,
                tickformat='%Y-%m-%d\n%H:%M',
                dtick=86400000.0  # 1 day in milliseconds
            ),
            yaxis=dict(
                gridcolor='lightgray',
                showgrid=True,
                tickfont=dict(size=10)  # Smaller font for longer labels
            ),
            legend=dict(
                title="Legend",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01
            ),
            margin=dict(l=300)  # Increase left margin for longer OPUP names
        )
    
    # Add annotations
    fig.add_annotation(
        text="<b>Schedulability Window</b>: OPUP can start anytime in this range<br>" +
             "<b>Scheduled Visit</b>: Actual scheduled visit execution<br>" +
             "<i>OPUPs ordered from earliest to latest start time</i>",
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=10),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    # Determine output path
    if output_dir is None:
        output_dir = Path(csv_filepath).parent
    else:
        output_dir = Path(output_dir)
    
    output_html = output_dir / f"opup_gantt_chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    # Save figure
    fig.write_html(str(output_html))
    
    print(f"\n{'='*60}")
    print("✅ Gantt Chart Generated!")
    print(f"{'='*60}")
    print(f"\n📊 Chart saved to: {output_html}")
    print(f"\nOpen in browser to view interactive chart\n")
    
    return str(output_html)

def create_detailed_opup_schedule(csv_filepath, output_dir=None):
    """
    Create a detailed multi-panel visualization showing:
    1. Overall OPUP timeline
    2. Visit-level details for each OPUP (with aligned time axes)
    3. Statistics summary
    
    Args:
        csv_filepath: Path to aggregated CSV file
        output_dir: Output directory for the chart
        
    Returns:
        Path to generated HTML visualization
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import datetime, timedelta
    from pathlib import Path
    
    print(f"\n{'='*60}")
    print("Creating Detailed OPUP Schedule Visualization")
    print(f"{'='*60}\n")
    
    # Load data
    df = pd.read_csv(csv_filepath)
    
    # Parse time function
    def parse_time_column(time_str):
        if pd.isna(time_str):
            return None
        time_str = str(time_str).strip().split()[0]
        try:
            parts = time_str.split('-')
            year = int(parts[0])
            doy = int(parts[1])
            if len(parts) > 2:
                time_parts = parts[2].split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                second = int(time_parts[2]) if len(time_parts) > 2 else 0
            else:
                hour = minute = second = 0
            dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second)
            return dt
        except:
            return None
    
    # Parse times
    df['Start_dt'] = df['Start'].apply(parse_time_column)
    df['Earliest_Start_dt'] = df['Earliest_Start_Time'].apply(parse_time_column)
    df['Latest_Start_dt'] = df['Latest_Start_Time'].apply(parse_time_column)
    df['Latest_End_dt'] = df['Latest_End_Time'].apply(parse_time_column)
    
    # Get unique OPUPs and sort by earliest Start time
    opup_start_times = {}
    for opup in df['OPUP_Source'].unique():
        opup_df = df[df['OPUP_Source'] == opup]
        min_start = opup_df['Start_dt'].min()
        opup_start_times[opup] = min_start if pd.notna(min_start) else datetime.max
    
    # Sort OPUPs by start time
    opups = sorted(opup_start_times.keys(), key=lambda x: opup_start_times[x])
    n_opups = len(opups)
    
    print(f"Found {n_opups} unique OPUP(s), ordered by start time")
    
    # Calculate global time range for alignment
    global_min_time = df['Earliest_Start_dt'].min()
    global_max_time = df['Latest_End_dt'].max()
    
    # Add some padding (5% on each side)
    time_range = global_max_time - global_min_time
    padding = time_range * 0.05
    global_min_time -= padding
    global_max_time += padding
    
    print(f"\nGlobal time range: {global_min_time.strftime('%Y-%m-%d %H:%M')} to {global_max_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Create OPUP display names with purposes
    opup_display_names = {}
    for opup in opups:
        opup_df = df[df['OPUP_Source'] == opup]
        opup_short = opup.replace('_opup', '').replace('.tgz', '')
        
        # Check if there's a single unique Intended_Purpose
        if 'Intended_Purpose' in df.columns:
            unique_purposes = opup_df['Intended_Purpose'].dropna().unique()
            if len(unique_purposes) == 1:
                purpose = unique_purposes[0]
                # Truncate if too long
                if len(purpose) > 40:
                    purpose = purpose[:37] + "..."
                opup_display_names[opup] = f"{opup_short}<br>{purpose}"
            else:
                opup_display_names[opup] = opup_short
        else:
            opup_display_names[opup] = opup_short
    
    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=n_opups + 1,
        cols=1,
        subplot_titles=['Overall OPUP Timeline (Chronological Order)'] + [opup_display_names[opup] for opup in opups],
        vertical_spacing=0.05,
        row_heights=[0.3] + [0.7/n_opups] * n_opups,
        shared_xaxes=True  # This aligns all x-axes
    )
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
              '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
    
    # Plot 1: Overall timeline
    for idx, opup in enumerate(opups):
        opup_df = df[df['OPUP_Source'] == opup]
        color = colors[idx % len(colors)]
        
        earliest = opup_df['Earliest_Start_dt'].min()
        latest_start = opup_df['Latest_Start_dt'].max()
        latest_end = opup_df['Latest_End_dt'].max()
        
        # Create short name for legend
        opup_short = opup.replace('_opup', '').replace('.tgz', '')
        
        # Schedulability window
        fig.add_trace(go.Scatter(
            x=[earliest, latest_start],
            y=[idx, idx],
            mode='lines+markers',
            name=f'{opup_short} - Window',
            line=dict(color=color, width=10),
            marker=dict(size=10, symbol='diamond'),
            showlegend=True,
            hovertext=f"Start: {opup_start_times[opup].strftime('%Y-%m-%d %H:%M')}"
        ), row=1, col=1)
        
        # Actual scheduled times
        for _, visit in opup_df.iterrows():
            if pd.notna(visit['Start_dt']):
                fig.add_trace(go.Scatter(
                    x=[visit['Start_dt']],
                    y=[idx],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='x'),
                    showlegend=False,
                    hovertext=f"Visit: {visit.get('Visit_ID', 'N/A')}"
                ), row=1, col=1)
    
    # Plot 2-N: Individual OPUP details (aligned to common time axis)
    for idx, opup in enumerate(opups):
        opup_df = df[df['OPUP_Source'] == opup].copy()
        opup_df = opup_df.sort_values('Start_dt')
        row_num = idx + 2  # Row 1 = overall, Row 2+ = individual OPUPs
        
        for visit_idx, (_, visit) in enumerate(opup_df.iterrows()):
            if pd.notna(visit['Start_dt']) and pd.notna(visit.get('Duration')):
                start = visit['Start_dt']
                end = start + timedelta(seconds=visit['Duration'])
                
                # Calculate y position - each visit gets its own row
                # Add small spacing between visits
                y_base = visit_idx * 1.2  # Increased spacing for clarity
                bar_height = 0.8
                
                # Visit bar
                fig.add_trace(go.Scatter(
                    x=[start, end, end, start, start],
                    y=[y_base, y_base, y_base + bar_height, y_base + bar_height, y_base],
                    fill='toself',
                    fillcolor=colors[idx % len(colors)],
                    line=dict(color='black', width=1),
                    mode='lines',
                    name=visit.get('Visit_ID', 'N/A'),
                    showlegend=False,
                    hovertext=f"Visit: {visit.get('Visit_ID', 'N/A')}<br>" +
                            f"Start: {start.strftime('%Y-%m-%d %H:%M:%S')}<br>" +
                            f"Filter: {visit.get('WFI_Optical_Element', 'N/A')}<br>" +
                            f"Duration: {visit.get('Duration', 'N/A')}s",
                    hoverinfo='text'
                ), row=row_num, col=1)
                
                # Schedulability window indicators (horizontal line showing window)
                if pd.notna(visit['Earliest_Start_dt']) and pd.notna(visit['Latest_Start_dt']):
                    fig.add_trace(go.Scatter(
                        x=[visit['Earliest_Start_dt'], visit['Latest_Start_dt']],
                        y=[y_base + bar_height/2, y_base + bar_height/2],  # Center of bar
                        mode='lines',
                        line=dict(color='orange', width=3, dash='dot'),
                        showlegend=False,
                        hovertext=f"Schedulability window: {visit['Earliest_Start_dt'].strftime('%Y-%m-%d %H:%M')} to {visit['Latest_Start_dt'].strftime('%Y-%m-%d %H:%M')}"
                    ), row=row_num, col=1)
                
                # Add vertical line at actual start time for clarity
                fig.add_trace(go.Scatter(
                    x=[start, start],
                    y=[y_base, y_base + bar_height],
                    mode='lines',
                    line=dict(color='darkgreen', width=2),
                    showlegend=False,
                    hovertext=f"Start: {start.strftime('%Y-%m-%d %H:%M:%S')}"
                ), row=row_num, col=1)
                
        # Update all x-axes to use the same range
        for i in range(1, n_opups + 2):
            fig.update_xaxes(
                range=[global_min_time, global_max_time],
                tickformat='%Y-%m-%d<br>%H:%M',
                showgrid=True,
                gridcolor='lightgray',
                row=i,
                col=1
            )
    
    # Update layout
    fig.update_layout(
        height=300 + n_opups * 250,  # Increased height for better visibility
        title_text="OPUP Detailed Schedule Analysis (Aligned Timeline)",
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date/Time (All plots aligned)", row=n_opups+1, col=1)
    fig.update_yaxes(title_text="OPUP Index", row=1, col=1)
    
    for i in range(n_opups):
        fig.update_yaxes(
            title_text="Visit #",
            showgrid=True,
            gridcolor='lightgray',
            row=i+2,
            col=1
        )
    
    # Save
    if output_dir is None:
        output_dir = Path(csv_filepath).parent
    else:
        output_dir = Path(output_dir)
    
    output_html = output_dir / f"opup_detailed_schedule_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
    fig.write_html(str(output_html))
    
    print(f"\n✅ Detailed schedule saved to: {output_html}")
    print(f"   All plots aligned to common time axis: {global_min_time.strftime('%Y-%m-%d %H:%M')} to {global_max_time.strftime('%Y-%m-%d %H:%M')}\n")
    
    return str(output_html)

def package_report_archive(opup_stem, output_dir, generated_files=None):
    """
    Bundle all generated report products into a single .tgz archive
    for easy sharing and portability.

    The archive preserves relative paths so that HTML cross-links
    (to sky_plots/, mosaic page, sky plotter, etc.) still work
    when extracted into a single folder.

    Args:
        opup_stem: str, the OPUP name stem (e.g. 'my_opup')
        output_dir: Path, directory containing all generated products
        generated_files: list of Path objects that were created
                         (if None, auto-discovers by stem prefix)

    Returns:
        Path to the created .tgz archive, or None on failure
    """
    output_dir = Path(output_dir)
    archive_name = output_dir / f"{opup_stem}_report_package.tgz"

    files_to_include = []

    if generated_files:
        # Use the explicit list — only include files that actually exist
        for fpath in generated_files:
            fpath = Path(fpath)
            if fpath.exists():
                files_to_include.append(fpath)
    else:
        # Fallback: auto-discover anything matching the opup_stem
        for fpath in sorted(output_dir.glob(f"{opup_stem}*")):
            if fpath.is_file() and fpath.name != archive_name.name:
                files_to_include.append(fpath)

    # Also include sky_plots/ PNGs directory if it exists
    sky_plots_dir = output_dir / "sky_plots"
    if sky_plots_dir.is_dir():
        for png_file in sorted(sky_plots_dir.glob("*.png")):
            files_to_include.append(png_file)

    if not files_to_include:
        print("  ⚠️  No report files found to package.")
        return None

    try:
        with tarfile.open(archive_name, 'w:gz') as tar:
            for fpath in files_to_include:
                arcname = fpath.relative_to(output_dir)
                tar.add(str(fpath), arcname=str(arcname))

        n_files = len(files_to_include)
        size_mb = archive_name.stat().st_size / (1024 * 1024)
        print(f"\n📦 Packaged {n_files} files into: {archive_name.name} ({size_mb:.1f} MB)")
        for fpath in files_to_include:
            rel = fpath.relative_to(output_dir)
            fsize = fpath.stat().st_size / 1024
            print(f"     {rel}  ({fsize:.0f} KB)")

        return archive_name

    except Exception as e:
        print(f"  ⚠️  Failed to create archive: {e}")
        return None
    

if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    
    # Extract args
    opup_filepaths = args.opup_filepath
    opup_directory = args.opup_directory
    scf_filepaths = args.scf_filepath
    visit_filepaths = args.visit_filepath
    output_dir = args.output_dir
    keep_GW = args.keep_GW
    output_format = args.format
    
    # If -opup_dir is provided, find all OPUP files in that directory
    directory_mode = False
    if opup_directory is not None:
        found_opups = find_opup_files_in_directory(opup_directory)
        # Append to any manually specified OPUP files
        opup_filepaths = list(opup_filepaths) + found_opups
        directory_mode = len(found_opups) > 0
    
    # Remove duplicates while preserving order
    if opup_filepaths:
        seen = set()
        opup_filepaths = [x for x in opup_filepaths if not (x in seen or seen.add(x))]
    
    # Process based on format
    if output_format == 'integrated':
        # Generate integrated report with sky plotter
        for opup_filepath in opup_filepaths:
            generate_integrated_report(opup_filepath, output_dir, keep_GW, generate_pngs=args.pngs)
        
        # If in directory mode, also generate aggregated output
        if directory_mode and len(opup_filepaths) > 1:
            print("\n" + "="*60)
            print("Directory mode: Generating aggregated analysis")
            print("="*60)
            aggregated_df, csv_path = aggregate_opup_dataframes(opup_filepaths, output_dir, keep_GW)
            # Generate Gantt charts if we have data
            if csv_path and not aggregated_df.empty:
                try:
                    print("\n" + "="*60)
                    print("Generating Gantt Charts")
                    print("="*60)
                    gantt_path = create_opup_gantt_chart(csv_path, output_dir)
                    detailed_path = create_detailed_opup_schedule(csv_path, output_dir)
                    
                    print("\n" + "="*60)
                    print("✅ All visualizations complete!")
                    print("="*60)
                    print(f"\n📊 Aggregated CSV:     {csv_path}")
                    print(f"📈 Gantt Chart:        {gantt_path}")
                    print(f"📉 Detailed Schedule:  {detailed_path}\n")
                except Exception as e:
                    print(f"\n⚠️  Error generating Gantt charts: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # Original workflow
        if output_format in ['csv', 'both']:
            # If in directory mode with multiple OPUPs, aggregate them
            if directory_mode and len(opup_filepaths) > 1:
                print("\n" + "="*60)
                print("Directory mode: Generating aggregated analysis")
                print("="*60)
                aggregated_df, csv_path, aggregate_opup_dataframes(opup_filepaths, output_dir, keep_GW)
            else:
                # Process individually
                process_OPUPs(opup_filepaths, output_dir, keep_GW)
            
            process_SCFs(scf_filepaths, output_dir, keep_GW)
            process_visits(visit_filepaths, output_dir, keep_GW)
        
        if output_format in ['html', 'both']:
            process_OPUPs_html(opup_filepaths, output_dir, keep_GW)


# %%
