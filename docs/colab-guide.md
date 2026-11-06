# Running on Google Colab from VS Code

Training runs on a Colab GPU through the [Google Colab extension for VS Code](https://marketplace.visualstudio.com/items?itemName=Google.colab). Local development and tests use the CPU.

## How a run works

Each run uses a self-contained notebook built from the current code:

```bash
python scripts/build_colab_notebook.py --job runtime-check --tag first-session
```

The notebook is written to `notebooks/`, which is not tracked by git. It embeds a compressed bundle of the package and study settings, protected by a SHA-256 checksum, so the runtime executes exactly the code that built it. When it runs, it:

1. mounts Google Drive and writes all outputs to `My Drive/compact-ser/`;
2. verifies and unpacks the embedded code;
3. starts the job as a background process, so stopping a notebook cell does not stop the job;
4. relays the job's log to the notebook;
5. shuts the runtime down after the job finishes, so it stops billing.

## Steps in VS Code

1. Open the notebook, then choose **Select Kernel → Colab → New Colab Server** and select a GPU.
2. Choose **Run All**. If Google Drive is not mounted, run **Colab: Mount Google Drive to Server...** from the Command Palette, then choose **Run All** again.
3. Leave the notebook running. VS Code saves the cell outputs to the local notebook file.
4. Check that the runtime no longer appears in Colab when the job ends. If a job fails, the notebook does not shut the runtime down, so disconnect it yourself.

## Jobs

| Job | Purpose | Data needed |
| --- | --- | --- |
| `runtime-check` | Records the environment, loads the pinned checkpoint, and measures training updates on synthetic audio | None |

## Records on Google Drive

| Path | Content |
| --- | --- |
| `<job>/environment/<timestamp>/` | Session details, `nvidia-smi` output, and installed packages |
| `<job>/job.log` | Console output from the job |
| `<job>/<job result>.json` | Measurements written by the job |
