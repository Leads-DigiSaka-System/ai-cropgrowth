"""Disk-backed rasters on one global grid, and COG mosaics.

Large outputs (a big province like Palawan, or the whole country) never fit in memory at 30 m:
the Philippines' bounding box is ~36 000 x 61 000 pixels. Everything here works block by block:

  GridWriter      a sparse, tiled, uncompressed GeoTIFF on disk (unwritten blocks cost nothing and
                  read back as nodata). Pieces (tiles, input blocks) are written into it where they
                  have data; .to_cog() then turns it into a compressed Cloud-Optimized GeoTIFF with
                  overviews. Peak memory: one piece + one block.
  mosaic_cogs()   merges many COGs (e.g. every province's stage map for one period) into one COG,
                  reading the inputs block by block.

All grids are snapped to multiples of the pixel size (grid_for_bbox), so every province product
lands on the same pixel grid and mosaics are exact copies, never resampled. Inputs that are not
on that grid (older files) are placed by nearest pixel centre.
"""
from __future__ import annotations

import math
import os
import tempfile
import threading

import numpy as np


# ------------------------------------------------------------------ grid
def grid_for_bbox(bbox, res):
    """(west, north, width, height) of the global res-grid covering bbox: pixel edges at
    integer multiples of res, so grids of different areas line up exactly."""
    w, s, e, n = bbox
    eps = res * 1e-6
    x0 = math.floor((w + eps) / res) * res
    y1 = math.ceil((n - eps) / res) * res
    x1 = math.ceil((e - eps) / res) * res
    y0 = math.floor((s + eps) / res) * res
    return x0, y1, max(1, int(round((x1 - x0) / res))), max(1, int(round((y1 - y0) / res)))


def grid_coords(bbox, res):
    """Pixel-centre coordinates (ys descending, xs) of grid_for_bbox(bbox, res)."""
    x0, y1, wpx, hpx = grid_for_bbox(bbox, res)
    xs = x0 + (np.arange(wpx) + 0.5) * res
    ys = y1 - (np.arange(hpx) + 0.5) * res
    return ys, xs


# ------------------------------------------------------------------ writer
class GridWriter:
    """Sparse on-disk raster (EPSG:4326) on the global grid over `bbox`.

    write(arrays, ys, xs): put a piece whose pixel centres are ys (rows) / xs (cols); only its
    valid pixels (!= nodata, finite) are written, and only where the raster is still nodata
    (first valid value wins, so overlapping pieces never overwrite each other).
    Thread-safe (tile workers write concurrently)."""

    def __init__(self, bbox, res, count=1, dtype="int16", nodata=-1, band_names=None, tags=None,
                 tmp_dir=None):
        import rasterio
        from rasterio.transform import from_origin
        self.res, self.count, self.dtype, self.nodata = float(res), int(count), np.dtype(dtype), nodata
        x0, y1, self.width, self.height = grid_for_bbox(bbox, self.res)
        self.x0, self.y1 = x0, y1
        self.transform = from_origin(x0, y1, self.res, self.res)
        fd, self.path = tempfile.mkstemp(suffix=".tif", prefix="grid_", dir=tmp_dir)
        os.close(fd)
        profile = dict(driver="GTiff", width=self.width, height=self.height, count=self.count,
                       dtype=self.dtype.name, crs="EPSG:4326", transform=self.transform, nodata=nodata,
                       tiled=True, blockxsize=512, blockysize=512, sparse_ok=True, bigtiff="IF_SAFER")
        self._ds = rasterio.open(self.path, "w+", **profile)
        for i, name in enumerate(band_names or [], 1):
            self._ds.set_band_description(i, str(name))
        if tags:
            self._ds.update_tags(**{k: str(v) for k, v in tags.items()})
        self._lock = threading.Lock()
        self.pixels_written = 0

    def _index(self, ys, xs):
        cols = np.floor((np.asarray(xs, float) - self.x0) / self.res).astype(np.int64)
        rows = np.floor((self.y1 - np.asarray(ys, float)) / self.res).astype(np.int64)
        return rows, cols

    def write(self, arrays, ys, xs):
        """arrays: (count, ny, nx) or (ny, nx) for count == 1."""
        from rasterio.windows import Window
        a = np.asarray(arrays)
        if a.ndim == 2:
            a = a[None]
        rows, cols = self._index(ys, xs)
        rk = (rows >= 0) & (rows < self.height)
        ck = (cols >= 0) & (cols < self.width)
        if not rk.any() or not ck.any():
            return 0
        a, rows, cols = a[:, rk][:, :, ck], rows[rk], cols[ck]
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        valid = np.zeros(a.shape[1:], bool)           # a pixel is data when any band is
        for b in a:
            ok = b != self.nodata
            if np.issubdtype(b.dtype, np.floating):
                ok &= np.isfinite(b)
            valid |= ok
        if not valid.any():
            return 0
        win = Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
        rr, cc = (rows - r0)[:, None], (cols - c0)[None, :]
        with self._lock:
            cur = self._ds.read(window=win)
            empty = (cur == self.nodata).all(axis=0)
            take = np.zeros(cur.shape[1:], bool)
            take[rr, cc] = valid
            take &= empty
            if not take.any():
                return 0
            src = np.full(cur.shape, self.nodata, dtype=self.dtype)
            src[:, rr, cc] = np.where(valid[None], a, self.nodata).astype(self.dtype)
            cur = np.where(take[None], src, cur)
            self._ds.write(cur, window=win)
            n = int(take.sum())
            self.pixels_written += n
            return n

    def to_cog(self, out_path, compress="DEFLATE", overview_resampling="nearest"):
        """Write the compressed COG (internal tiles + overviews) and delete the temp file."""
        from rasterio.shutil import copy as rio_copy
        self._ds.close()
        try:
            rio_copy(self.path, out_path, driver="COG", compress=compress, blocksize=512,
                     overview_resampling=overview_resampling, bigtiff="IF_SAFER", num_threads="ALL_CPUS")
        finally:
            self.discard()
        return out_path

    def discard(self):
        try:
            if not self._ds.closed:
                self._ds.close()
        except Exception:
            pass
        if os.path.exists(self.path):
            os.remove(self.path)


# ------------------------------------------------------------------ mosaics of existing COGs
def mosaic_cogs(paths, out_path, res=None, bbox=None, nodata=None, log=print):
    """
    Merge rasters (EPSG:4326, same band count / dtype) into one COG, reading block by block.
    Where inputs overlap, the first input with data wins. res / bbox default to the finest
    input pixel size and the union of the inputs, snapped to the global grid.
    Returns {'path', 'inputs', 'width', 'height', 'pixels_with_data'}.
    """
    import rasterio
    paths = [p for p in paths if p]
    if not paths:
        raise ValueError("nothing to mosaic")
    meta = []
    for p in paths:
        with rasterio.open(p) as src:
            if src.crs is None or src.crs.to_epsg() != 4326:
                raise ValueError(f"{p}: expected EPSG:4326, got {src.crs}")
            meta.append(dict(bounds=src.bounds, res=abs(src.transform.a), count=src.count,
                             dtype=src.dtypes[0], nodata=src.nodata,
                             names=src.descriptions, tags=src.tags()))
    counts, dtypes = {m["count"] for m in meta}, {m["dtype"] for m in meta}
    if len(counts) > 1 or len(dtypes) > 1:
        raise ValueError(f"inputs differ in band count {counts} or type {dtypes}")
    res = res or min(m["res"] for m in meta)
    if bbox is None:
        bbox = (min(m["bounds"].left for m in meta), min(m["bounds"].bottom for m in meta),
                max(m["bounds"].right for m in meta), max(m["bounds"].top for m in meta))
    nodata = meta[0]["nodata"] if nodata is None else nodata
    if nodata is None:
        raise ValueError("inputs have no nodata value; pass nodata=")
    tags = dict(meta[0]["tags"]); tags["mosaic_of"] = str(len(paths))
    writer = GridWriter(bbox, res, count=meta[0]["count"], dtype=meta[0]["dtype"], nodata=nodata,
                        band_names=[n for n in meta[0]["names"] if n], tags=tags)
    try:
        for i, p in enumerate(paths, 1):
            with rasterio.open(p) as src:
                t = src.transform
                for _, win in src.block_windows(1):
                    data = src.read(window=win)
                    if src.nodata is not None and src.nodata != nodata:
                        data = np.where(data == src.nodata, nodata, data)
                    cols = win.col_off + np.arange(win.width) + 0.5
                    rows = win.row_off + np.arange(win.height) + 0.5
                    xs = t.c + cols * t.a
                    ys = t.f + rows * t.e
                    writer.write(data, ys, xs)
            if log and (i % 10 == 0 or i == len(paths)):
                log(f"  mosaic: {i}/{len(paths)} inputs")
        n = writer.pixels_written
        w, h = writer.width, writer.height
        writer.to_cog(out_path)
    except BaseException:
        writer.discard()
        raise
    return {"path": out_path, "inputs": len(paths), "width": w, "height": h, "pixels_with_data": n}
