import os
import re
import math
import fitz


def _collect_files(folder, pattern=None, recursive=False):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    regex = re.compile(pattern) if pattern else None
    exts = {".pdf"}
    files = []

    if recursive:
        for root, _, filenames in os.walk(folder):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext not in exts:
                    continue
                if regex and not regex.search(name):
                    continue
                files.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            full_path = os.path.join(folder, name)
            if not os.path.isfile(full_path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue
            if regex and not regex.search(name):
                continue
            files.append(full_path)

    files.sort()
    return files


def combine_pdfs_grid(
    paths,
    rows,
    cols,
    out_path,
    align="top",
    h_align="left",
):
    docs = []
    pages = []
    sizes = []

    try:
        for path in paths:
            doc = fitz.open(path)
            page = doc.load_page(0)
            rect = page.rect
            docs.append(doc)
            pages.append(page)
            sizes.append((rect.width, rect.height))

        col_widths = [0.0] * cols
        row_heights = [0.0] * rows

        for idx, (w, h) in enumerate(sizes):
            r = idx // cols
            c = idx % cols
            if r >= rows:
                break
            if w > col_widths[c]:
                col_widths[c] = w
            if h > row_heights[r]:
                row_heights[r] = h

        total_width = sum(col_widths)
        total_height = sum(row_heights)

        out_doc = fitz.open()
        out_page = out_doc.new_page(width=total_width, height=total_height)

        x_offsets = [0.0] * cols
        y_offsets = [0.0] * rows
        for c in range(1, cols):
            x_offsets[c] = x_offsets[c - 1] + col_widths[c - 1]
        for r in range(1, rows):
            y_offsets[r] = y_offsets[r - 1] + row_heights[r - 1]

        for idx, (page, (w, h)) in enumerate(zip(pages, sizes)):
            r = idx // cols
            c = idx % cols
            if r >= rows:
                break

            cell_x = x_offsets[c]
            cell_y = y_offsets[r]
            cell_w = col_widths[c]
            cell_h = row_heights[r]

            if h_align == "right":
                dx = cell_w - w
            elif h_align == "center":
                dx = (cell_w - w) / 2
            else:
                dx = 0

            if align == "bottom":
                dy = cell_h - h
            elif align == "center":
                dy = (cell_h - h) / 2
            else:
                dy = 0

            rect = fitz.Rect(cell_x + dx, cell_y + dy, cell_x + dx + w, cell_y + dy + h)
            out_page.show_pdf_page(rect, docs[idx], 0)

        out_doc.save(out_path)
        out_doc.close()
    finally:
        for doc in docs:
            doc.close()


def compare_plots(
    folder,
    pattern=None,
    shape=None,
    recursive=False,
    save_path=None,
    align="top",
    h_align="left",
):
    """
    Create a vector PDF grid from a folder of PDFs.

    Args:
        folder (str): Folder containing plot files.
        pattern (str|None): Regex pattern to filter files by filename.
        shape (tuple|None): (rows, cols). If None, choose a square-ish layout.
        recursive (bool): Search subfolders as well.
        save_path (str): Output PDF path.
        align (str): Vertical alignment for vector PDF cells: top, center, bottom.
        h_align (str): Horizontal alignment for vector PDF cells: left, center, right.
    """
    files = _collect_files(folder, pattern=pattern, recursive=recursive)
    if not files:
        raise ValueError("No plot files matched the criteria.")

    n = len(files)
    if shape is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
    else:
        rows, cols = shape
        if rows * cols < n:
            raise ValueError(
                f"Shape {shape} too small for {n} plots. Increase rows/cols."
            )

    if save_path is None:
        raise ValueError("save_path is required.")
    if not save_path.lower().endswith(".pdf"):
        raise ValueError("save_path must end with .pdf.")

    combine_pdfs_grid(files, rows, cols, save_path, align=align, h_align=h_align)
    return save_path


def combine_pdf_cdf_plots(
    folder,
    recursive=False,
    prefer_ccdf=True,
    output_folder="combinedPlots",
    align="top",
):
    """
    Find matching PDF/CDF (or CCDF) plots and save side-by-side comparisons.

    Filenames are matched by stripping the _pdf/_cdf/_ccdf suffix (before .pdf).
    Combines PDF pages without rasterizing (requires PyMuPDF).
    """
    files = _collect_files(folder, pattern=r"\.pdf$", recursive=recursive)
    if not files:
        raise ValueError("No PDF files found in the folder.")

    by_base = {}
    for path in files:
        name = os.path.basename(path)
        if name.endswith("_pdf.pdf"):
            base = name[: -len("_pdf.pdf")]
            by_base.setdefault(base, {})["pdf"] = path
        elif name.endswith("_cdf.pdf"):
            base = name[: -len("_cdf.pdf")]
            by_base.setdefault(base, {})["cdf"] = path
        elif name.endswith("_ccdf.pdf"):
            base = name[: -len("_ccdf.pdf")]
            by_base.setdefault(base, {})["ccdf"] = path

    out_dir = os.path.join(folder, output_folder)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for base, paths in by_base.items():
        if "pdf" not in paths:
            continue
        if prefer_ccdf and "ccdf" in paths:
            cdf_path = paths["ccdf"]
            cdf_label = "CCDF"
        elif "cdf" in paths:
            cdf_path = paths["cdf"]
            cdf_label = "CDF"
        elif "ccdf" in paths:
            cdf_path = paths["ccdf"]
            cdf_label = "CCDF"
        else:
            continue

        out_path = os.path.join(out_dir, f"{base}_pdf_vs_{cdf_label.lower()}.pdf")
        combine_pdfs_grid(
            [paths["pdf"], cdf_path], 1, 2, out_path, align=align, h_align="left"
        )
        saved += 1

    return saved


if __name__ == "__main__":
    folder_path = "/Users/eliaslundheim/work/PhD/SimulationScripts/Plots/powerLaw"
    nr_saved = combine_pdf_cdf_plots(folder_path, prefer_ccdf=True)
    print(f"Saved {nr_saved} combined plots to combinedPlots")
