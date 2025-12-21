import os
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def post_pdf(post_path, internal_mesh):
    """
    Generate a comprehensive PDF report from post-processing results.
    Includes summary statistics calculated from a VTK mesh and images for all existing slices.
    
    Args:
        post_path (str): Path to post-processing output directory containing images.
        internal_mesh (pyvista.UnstructuredGrid or PolyData): CFD results mesh.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"    * Generating PDF report from: {post_path}")

    # Output PDF filename
    pdf_filename = os.path.join(post_path, "post_report.pdf")
    logger.info(f"    * Creating PDF report: {pdf_filename}")

    # PDF canvas
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    width, height = A4

    # --------------------------------------------------------
    # Title page
    # --------------------------------------------------------
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 100, "CFD Post-Processing Report")
    c.setFont("Helvetica", 14)
    c.drawCentredString(width / 2, height - 140, "Orthogonal Slices of CFD Fields")
    c.setFont("Helvetica-Oblique", 12)
    c.drawCentredString(width / 2, height - 180, "Automatically generated from CFD VTK data")
    c.showPage()

    # --------------------------------------------------------
    # Collect all existing images
    # --------------------------------------------------------
    img_dir = os.path.join(post_path, "images")
    if not os.path.exists(img_dir):
        logger.error(f"    * Image directory not found: {img_dir}")
        return

    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    if not all_images:
        logger.warning("    * No images found for report.")
        return

    # Group images by variable
    images_by_var = {}
    for fname in all_images:
        var_name = fname.split("_slice_")[0]
        images_by_var.setdefault(var_name, []).append(fname)

    # --------------------------------------------------------
    # Add sections for each variable
    # --------------------------------------------------------
    for var_name, fnames in images_by_var.items():
        logger.info(f"    * Adding section for variable: {var_name}")

        # --------------------------------------------------------
        # Compute summary statistics from VTK mesh if available
        # --------------------------------------------------------
        if var_name in internal_mesh.point_data:
            data_array = internal_mesh.point_data[var_name]
            var_min = float(np.nanmin(data_array))
            var_max = float(np.nanmax(data_array))
            var_mean = float(np.nanmean(data_array))
            var_std = float(np.nanstd(data_array))
        else:
            var_min = var_max = var_mean = var_std = np.nan
            logger.warning(f"    * Variable {var_name} not found in mesh. Statistics set to NaN.")

        # -----------------------------
        # Summary statistics page
        # -----------------------------
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(width / 2, height - 100, f"Variable: {var_name}")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 150, f"Summary statistics for '{var_name}':")
        c.drawString(70, height - 170, f"Minimum value: {var_min:.4f}")
        c.drawString(70, height - 190, f"Maximum value: {var_max:.4f}")
        c.drawString(70, height - 210, f"Mean value: {var_mean:.4f}")
        c.drawString(70, height - 230, f"Standard deviation: {var_std:.4f}")
        c.showPage()

        # -----------------------------
        # Add slices (existing images only)
        # -----------------------------
        slices_per_page = 2
        slice_spacing = 20
        img_max_width = width - 100
        img_max_height = (height - 150 - slice_spacing * slices_per_page) / slices_per_page

        for i in range(0, len(fnames), slices_per_page):
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(width / 2, height - 50, f"{var_name} Slices (Z-direction)")

            y_pos = height - 100
            for j in range(slices_per_page):
                idx = i + j
                if idx >= len(fnames):
                    break

                fname = fnames[idx]
                img_path = os.path.join(img_dir, fname)
                if not os.path.exists(img_path):
                    logger.warning(f"    * Image not found: {img_path}")
                    continue

                # Open and resize image maintaining aspect ratio
                img = Image.open(img_path)
                img_width, img_height = img.size
                aspect = img_height / img_width

                new_width = min(img_max_width, img_width)
                new_height = min(img_max_height, new_width * aspect)
                if new_height > img_max_height:
                    new_height = img_max_height
                    new_width = new_height / aspect

                x_pos = (width - new_width) / 2
                y_pos -= new_height

                c.drawImage(ImageReader(img), x_pos, y_pos, width=new_width, height=new_height)

                # Caption below each image
                c.setFont("Helvetica", 10)
                c.drawCentredString(width / 2, y_pos - 12,
                                    f"Slice {idx+1}: Orthogonal cut through {var_name} at Z-plane {idx+1}/{len(fnames)}")

                y_pos -= (slice_spacing + 12)

            c.showPage()

    # --------------------------------------------------------
    # Save PDF
    # --------------------------------------------------------
    c.save()
    logger.info(f"    * PDF report generated successfully: {pdf_filename}")
