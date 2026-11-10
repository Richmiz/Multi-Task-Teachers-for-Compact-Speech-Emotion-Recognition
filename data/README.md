# Data

Datasets are not stored in this repository, and each one remains under its original licence. Place local copies in this folder using the names below.

| Dataset | Role | Folder | Access |
| --- | --- | --- | --- |
| IEMOCAP | Training, validation, and held-out testing | `data/iemocap/` | Request a licence from [USC SAIL](https://sail.usc.edu/iemocap/iemocap_release.htm) |
| RAVDESS | External test | `data/ravdess/` | [Zenodo record 1188976](https://zenodo.org/records/1188976) |
| Berlin Emo-DB | External test | `data/emodb/` | [TU Berlin](https://www.tu.berlin/kw/forschung/projekte/emotionale-sprechweise/) |

On Google Colab, IEMOCAP is stored and restored with the `iemocap-store` and `iemocap-check` jobs described in [the Colab guide](../docs/colab-guide.md), which also shows how to build the stored copy from an archive that was already downloaded. Build manifests with `scripts/build_manifests.py`. They are written to `data/manifests/`, which git ignores. The IEMOCAP folder should contain `Session1` to `Session5`.

IEMOCAP audio, labels, transcripts, and anything derived from them must not be committed or shared unless the licensor permits it. External test data must not be used for development, tuning, or debugging.
