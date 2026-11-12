# Running on Google Colab from VS Code

Training runs on a Colab GPU through the [Google Colab extension for VS Code](https://marketplace.visualstudio.com/items?itemName=Google.colab). Local development and tests use the CPU.

## How a run works

Every Colab job runs from one notebook, [`notebooks/colab_runner.ipynb`](../notebooks/colab_runner.ipynb). Its first cell names the job, its options, and the commit to run. When it runs, it:

1. fetches this repository from GitHub at that commit;
2. mounts Google Drive and writes all outputs to `My Drive/Research/compact-ser/`;
3. starts the job as a background process, so stopping a notebook cell does not stop the job;
4. relays the job's log to the notebook and saves it on Drive with the commit it ran;
5. shuts the runtime down after the job completes, so it stops billing.

Colab runs only code that has been pushed, so every result on Drive can be traced to a commit. Every training result also records that commit.

## Steps in VS Code

1. Commit and push the code to run.
2. Open the notebook, then choose **Select Kernel → Colab → New Colab Server**. Select a GPU, or a CPU runtime for the IEMOCAP jobs.
3. Set `JOB`, `JOB_OPTIONS`, and `CODE_REF` in the first cell, then choose **Run All**. If Google Drive is not mounted, run **Colab: Mount Google Drive to Server...** from the Command Palette, then choose **Run All** again.
4. Leave the notebook running. If a job fails, the notebook does not shut the runtime down, so disconnect it yourself.
5. Clear the notebook's outputs before committing any change to it.

## Jobs

| `JOB` | Purpose | Runtime | Outputs in `My Drive/Research/compact-ser/` |
| --- | --- | --- | --- |
| `runtime-check` | Records the environment, loads the pinned checkpoint, and measures training updates on synthetic audio | GPU | `runtime_check/` |
| `iemocap-store` | Downloads the licensed IEMOCAP archive once, keeps only the files the study uses, checks them, and saves the verified copy to Drive | CPU | `datasets/iemocap/`, `iemocap_store/` |
| `iemocap-check` | Copies the stored copy from Drive to the runtime, verifies it, and checks the four-class counts | CPU | `iemocap_check/` |
| `training` | Restores IEMOCAP the same way, then executes the next pending planned runs | GPU | `study/` |

`training` takes these options:

| Option | Effect |
| --- | --- |
| `--scope pilot`, `core`, or `full` | Plan to follow: `pilot` is the validation pilot (teacher tuning and pilot students), `core` adds the primary comparison, and `full` adds every other run |
| `--max-runs N` | Execute at most `N` runs in this session |
| `--runs RUN_ID ...` | Execute only the named planned runs, in plan order |

The run ledger in `study/` records each run's progress, so the next session continues with the next pending run. A run interrupted by a lost runtime starts again from the beginning. A lock file stops two runtimes from training the same study at once; a lock that has not been refreshed for 10 minutes is taken over.

`study/study_settings.json` records the training settings of the runs in the folder. If the settings in `configs/study.toml` change, the job stops before training, so that runs trained with different settings are never mixed. Rename the old folder on Drive, for example to `pilot-1`, and the next run starts a new `study/` folder.

The log shows validation UAR, and word error rate for teachers that learn transcription, at every validation pass, so a run that is not learning can be stopped early.

## Data on Google Drive

IEMOCAP is downloaded with a temporary credential, so it cannot be downloaded again in each session as public datasets can. The `iemocap-store` job therefore saves one verified copy:

```
My Drive/Research/compact-ser/datasets/iemocap/
  iemocap_study_files.tar    utterance audio, emotion evaluations, and transcripts only
  iemocap_study_files.json   checksums of the archive and every file, and of the original release
```

The stored copy holds only the files the study uses, so each session reads much less from Drive than the full release. Copies to and from Drive are made in chunks and resumed after an interrupted connection, and every copy is checked against its SHA-256 checksum before use. Pass the download details supplied by USC as the job's options, and do not commit them:

```python
JOB = "iemocap-store"
JOB_OPTIONS = ["--source-url", URL, "--username", NAME, "--expected-md5", MD5]
```

The password is requested when the notebook runs and is passed to the job without being saved.

If the release archive was already downloaded to a local computer, build the stored copy there instead and upload its two files to `My Drive/Research/compact-ser/datasets/iemocap/`:

```bash
python -m compact_ser.iemocap_archive store --source-archive PATH/TO/IEMOCAP_full_release.tar.gz \
    --work-dir data/iemocap-work --store-dir data/iemocap-upload --output-dir data/iemocap-work/logs
```

The archive is read to its end, so a truncated or damaged download is detected, and the four-class counts are checked before anything is written to the upload folder. Then run the `iemocap-check` job to confirm that a Colab runtime can restore the uploaded copy. If the two files are not in `My Drive/Research/compact-ser/datasets/iemocap/`, the IEMOCAP and training jobs search My Drive for them and report the folder they used.

## Records on Google Drive

| Path | Content |
| --- | --- |
| `<job>/job.log` | Console output from every run of the job, each headed by its start time and commit |
| `<job>/environment/<timestamp>/` | Session details, `nvidia-smi` output, and installed packages |
| `<job>/<job result>.json` | Measurements written by the job |
| `study/run_ledger.jsonl` | Start, completion, and failure of every training run |
| `study/runs/<run id>/` | Each run's selected model and its result, including validation metrics and the commit |
