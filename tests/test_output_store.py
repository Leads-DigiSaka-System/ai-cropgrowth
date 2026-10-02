import pytest

from cropgrowth_agent import output_store as st


class _Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def upload_from_filename(self, f):
        self.bucket.data[self.name] = open(f).read()

    def exists(self):
        self.bucket.exists_calls += 1
        return self.name in self.bucket.data

    def download_as_text(self):
        return self.bucket.data[self.name]


class _Bucket:
    def __init__(self):
        self.data, self.exists_calls = {}, 0

    def blob(self, name):
        return _Blob(self, name)


class FakeGCS:
    def __init__(self):
        self.b = _Bucket()

    def bucket(self, name):
        return self.b

    def list_blobs(self, bucket, prefix=""):
        return [_Blob(self.b, n) for n in self.b.data if n.startswith(prefix)]


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hi")
    return str(p)


def test_local_store(tmp_path, src):
    s = st.LocalStore(tmp_path / "out")
    assert s.put(src, "p/x/a.txt").endswith("out/p/x/a.txt")
    assert s.exists("p/x/a.txt") and s.list("p/") == ["p/x/a.txt"] and s.read_text("p/x/a.txt") == "hi"


def test_gcs_store_uses_one_listing(src):
    client = FakeGCS()
    g = st.GCSStore(client, "bkt")
    assert g.put(src, "p/a.txt") == "gs://bkt/p/a.txt"
    g.refresh("p/")
    assert g.exists("p/a.txt") and not g.exists("p/b.txt")
    assert client.b.exists_calls == 0
    assert g.read_text("p/a.txt") == "hi"


def test_multi_store(tmp_path, src):
    g, l = st.GCSStore(FakeGCS(), "bkt"), st.LocalStore(tmp_path / "out")
    m = st.MultiStore([g, l])
    m.put(src, "p/m.txt")
    assert g.exists("p/m.txt") and l.exists("p/m.txt") and m.exists("p/m.txt")
    assert m.describe() == f"gcs:gs://bkt + local:{l.root}"


def test_drive_needs_mount_and_gcs_needs_client():
    with pytest.raises(RuntimeError, match="not mounted"):
        st.DriveStore("/content/drive/MyDrive/x")
    with pytest.raises(ValueError):
        st.make_store("gcs")
    with pytest.raises(ValueError):
        st.make_store("dropbox")
