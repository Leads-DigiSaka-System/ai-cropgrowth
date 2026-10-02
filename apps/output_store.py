"""
output_store.py
==================================================================
Where products are saved. Every store takes a local file and a relative
path ("products/growth_stage/2026/dry/dry2026_BOHOL_202602.tiff") and puts
it in its destination:

  GCSStore    gs://<bucket>/<path>                (google-cloud-storage client)
  DriveStore  <Drive folder>/<path>               (Google Drive mounted in Colab
                                                   or Google Drive for desktop)
  LocalStore  <folder>/<path>                     (any local / network folder)
  MultiStore  several of the above at once        (e.g. GCS + Drive)

make_store('gcs' | 'gdrive' | 'both' | 'local', ...) builds one from settings.
==================================================================
"""

import os
import shutil

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/AI-CropGrowth/outputs"


class LocalStore:
    """Copies files under a root folder."""
    kind = "local"

    def __init__(self, root):
        self.root = os.path.abspath(os.path.expanduser(root))

    def _full(self, rel):
        return os.path.join(self.root, rel.lstrip("/"))

    def uri(self, rel):
        return self._full(rel)

    def put(self, local_path, rel):
        dst = self._full(rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        shutil.copyfile(local_path, tmp)            # copy then rename: no half-written files
        os.replace(tmp, dst)
        return self.uri(rel)

    def exists(self, rel):
        return os.path.exists(self._full(rel))

    def list(self, prefix=""):
        base = self._full(prefix)
        top = base if os.path.isdir(base) else os.path.dirname(base)
        out = []
        for d, _, files in os.walk(top):
            for f in files:
                full = os.path.join(d, f)
                if full.startswith(base) and not f.endswith(".part"):
                    out.append(os.path.relpath(full, self.root))
        return sorted(out)

    def read_text(self, rel):
        with open(self._full(rel), encoding="utf-8") as fh:
            return fh.read()

    def refresh(self, prefix=""):
        pass

    def describe(self):
        return f"{self.kind}:{self.root}"


class DriveStore(LocalStore):
    """Google Drive through its mounted folder (Colab: drive.mount('/content/drive'))."""
    kind = "gdrive"

    def __init__(self, root=DEFAULT_DRIVE_ROOT):
        super().__init__(root)
        mount = "/content/drive"
        if self.root.startswith(mount) and not os.path.isdir(os.path.join(mount, "MyDrive")):
            raise RuntimeError("Google Drive is not mounted: run "
                               "`from google.colab import drive; drive.mount('/content/drive')` first")


class GCSStore:
    """Google Cloud Storage bucket. `exists` uses one cached listing per
    refresh(prefix) instead of a request per file."""
    kind = "gcs"

    def __init__(self, client, bucket):
        self.client, self.bucket_name = client, bucket
        self.bucket = client.bucket(bucket)
        self._cache, self._cached_prefixes = set(), []

    def uri(self, rel):
        return f"gs://{self.bucket_name}/{rel.lstrip('/')}"

    def put(self, local_path, rel):
        rel = rel.lstrip("/")
        self.bucket.blob(rel).upload_from_filename(local_path)
        self._cache.add(rel)
        return self.uri(rel)

    def refresh(self, prefix=""):
        prefix = prefix.lstrip("/")
        names = {b.name for b in self.client.list_blobs(self.bucket_name, prefix=prefix)}
        self._cache = {n for n in self._cache if not n.startswith(prefix)} | names
        if prefix not in self._cached_prefixes:
            self._cached_prefixes.append(prefix)

    def exists(self, rel):
        rel = rel.lstrip("/")
        if any(rel.startswith(p) for p in self._cached_prefixes):
            return rel in self._cache
        return self.bucket.blob(rel).exists()

    def list(self, prefix=""):
        prefix = prefix.lstrip("/")
        return sorted(b.name for b in self.client.list_blobs(self.bucket_name, prefix=prefix))

    def read_text(self, rel):
        return self.bucket.blob(rel.lstrip("/")).download_as_text()

    def describe(self):
        return f"gcs:gs://{self.bucket_name}"


class MultiStore:
    """Writes to every store; reads from the first. A file counts as
    existing only when every store has it."""
    kind = "multi"

    def __init__(self, stores):
        if not stores:
            raise ValueError("MultiStore needs at least one store")
        self.stores = list(stores)

    def uri(self, rel):
        return self.stores[0].uri(rel)

    def put(self, local_path, rel):
        return [s.put(local_path, rel) for s in self.stores][0]

    def uris(self, rel):
        return [s.uri(rel) for s in self.stores]

    def exists(self, rel):
        return all(s.exists(rel) for s in self.stores)

    def list(self, prefix=""):
        return self.stores[0].list(prefix)

    def read_text(self, rel):
        return self.stores[0].read_text(rel)

    def refresh(self, prefix=""):
        for s in self.stores:
            s.refresh(prefix)

    def describe(self):
        return " + ".join(s.describe() for s in self.stores)


TARGETS = ("gcs", "gdrive", "both", "local")


def make_store(target, gcs_client=None, gcs_bucket=None, drive_root=DEFAULT_DRIVE_ROOT,
               local_root="outputs"):
    """
    target : 'gcs'    -> GCSStore(gcs_client, gcs_bucket)
             'gdrive' -> DriveStore(drive_root)
             'both'   -> GCS + Drive
             'local'  -> LocalStore(local_root)
    """
    if target not in TARGETS:
        raise ValueError(f"output target must be one of {TARGETS}")

    def gcs():
        if gcs_client is None or not gcs_bucket:
            raise ValueError("GCS output needs gcs_client and gcs_bucket")
        return GCSStore(gcs_client, gcs_bucket)

    if target == "gcs":
        return gcs()
    if target == "gdrive":
        return DriveStore(drive_root)
    if target == "both":
        return MultiStore([gcs(), DriveStore(drive_root)])
    return LocalStore(local_root)
